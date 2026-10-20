import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"


import pickle
from transformers import set_seed, AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig
from transformers import DataCollatorForLanguageModeling
import torch
from utils import (
    encode_dataset, 
    get_tokenizer, 
    get_data, 
    get_args,
    get_selected_token_ids,
    GenereteTextCallback,
    GrammarCallback,
    compute_metrics,
    preprocess_logits,
    compute_inference_results,
    process_for_under_trained_tokens,
    text_generation,
    min_distant_sequences,
    NoShuffleTrainer,
)
import json
import pandas as pd
import wandb
import logging
import time
from datetime import datetime
import accelerate
from torch.utils.data import Subset
from config import model_offloading, lr_dict, batch_size_dict



def training(args, dataset_dict, max_sequence_length, unique_tokens, train_test_distance):

    set_seed(args.run_seed)
    tokenizer, checkpoint_path = get_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id

    if not args.nlp_dataset:
        selected_token_ids = get_selected_token_ids(tokenizer, unique_tokens)
    else:
        selected_token_ids = [tokenizer.pad_token_id]
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError):
        assert args.incontext_input
        local_rank = 0

    deepspeed_config = "additional/deepspeed_config.json" if args.use_deepspeed and args.model_name not in model_offloading else (
                  "additional/deepspeed_config_offloading.json" if args.use_deepspeed else None
    )        
    
    
    # tokenize
    if not args.nlp_dataset:
        # apply character level tokenization
        encoded_dataset  = encode_dataset(tokenizer=tokenizer, 
                                        dataset=dataset_dict, 
                                        max_sequence_length=max_sequence_length, 
                                        verbose=True if local_rank == 0 else False,
                                        instruction_data={
                                            "add_instruction": args.add_instruction and (not args.nlp_dataset) and args.incontext_input,
                                            "instruction": args.instruction
                                        }
        )
    else:
        # NLP dataset where tokenization is standard
        # first tokenize to determine max length, then tokenize with padding to max length
        
        def tokenize_no_trunc(example):
            return tokenizer(example["text"])
        
        encoded_dataset_no_trunc = dataset_dict.map(
            tokenize_no_trunc
        )

        lengths = []
        for eval_dataset in encoded_dataset_no_trunc:
            lengths += [len(x) for x in encoded_dataset_no_trunc[eval_dataset]["input_ids"]]
        
        max_length = max(lengths)
        print(f"Max length: {max_length}")

        def tokenize_function(example):
            if tokenizer.padding_side == "right":
                tokenizer.padding_side = "left"         
            return tokenizer(example["text"], 
                             padding="max_length", 
                             truncation=True,
                             return_token_type_ids=False,
                             max_length=max_length+1
                             )

        
        encoded_dataset = dataset_dict.map(
            tokenize_function,
            batched=True
        )

        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text"])

        
    dataset = encoded_dataset.remove_columns(["text"])
    
    if args.use_under_trained_tokens and local_rank == 0:
        dataset, selected_token_ids = process_for_under_trained_tokens(args, tokenizer, dataset, selected_token_ids)
    
    
    
    if local_rank == 0:
        print(dataset)
        for eval_dataset in dataset:
            print(eval_dataset)
            print(dataset[eval_dataset]["input_ids"][0])
            print(dataset[eval_dataset]["input_ids"].shape)
        
        
    

    current_time = datetime.now()

    save_strategy = args.evaluation_strategy if args.save_checkpoint or args.save_best_model else 'no'
    if args.incontext_input:
        save_strategy = 'no'

    # output directory (initially stored in /tmp, and later moved to ./artifacts/)
    incontext_common_prefix_len = None
    if args.incontext_input:
        if args.considered_incontext_examples > 0:
            assert "incontext_common_prefix" in dataset
            assert dataset['incontext_common_prefix']['input_ids'].shape[0] > 0
            incontext_common_prefix_len = len([token for token in dataset['incontext_common_prefix']['input_ids'][0] if token not in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]])
        else:
            incontext_common_prefix_len = 0
        args.considered_incontext_examples = (args.considered_incontext_examples, incontext_common_prefix_len) # update
        output_dir = f"/tmp/incontext_{current_time.strftime('%Y_%m_%d_%H_%M')}_{args.model_name.replace('/', '_')}_{args.grammar_name}_{args.num_samples}_{args.run_seed}_{args.comment.replace(' ', '_')}_{args.considered_incontext_examples[0]}"

        if local_rank == 0:
            print(f"Using incontext common prefix len: {incontext_common_prefix_len}")
            
    else:
        output_dir = f"/tmp/output_{current_time.strftime('%Y_%m_%d_%H_%M')}_{args.model_name.replace('/', '_')}_{args.grammar_name}_{args.num_samples}_{args.run_seed}_{args.comment.replace(' ', '_')}_{args.considered_training_samples}"


    if local_rank == 0:
        print()
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print(f"Selected token ids: {selected_token_ids}")
        print("Deepspeed config file: ", deepspeed_config)
        print()
    
    # store args
    if(local_rank == 0):
        os.makedirs(output_dir, exist_ok=True)    
        with open(os.path.join(output_dir, "args.pkl"), "wb") as f:
            pickle.dump(vars(args), f)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    
    
    run_name = f"gl | {args.model_name} | {args.grammar_name} | {args.num_samples} | {current_time.strftime('%Y_%m_%d_%H_%M')}"

    # for wandb
    params_dict = {
        'max_sequence_length': max_sequence_length,
        'lr_scheduler': args.lr_scheduler,
        'warmup_ratio' : args.warmup_ratio,
        'output_dir': output_dir,
    }

    for key, value in vars(args).items():
        params_dict[key] = value

    if(local_rank == 0):
        os.environ["WANDB_WATCH"] = "all"
        os.environ["WANDB_API_KEY"]=os.getenv("WANDB_API_KEY", "")
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb_project_name = "diff_training_modes_acl"
        os.environ["WANDB_PROJECT"] = wandb_project_name
        wandb.init(project=wandb_project_name,
                    dir=output_dir, 
                    group=run_name)
        wandb.run.name = run_name
        wandb.config.update(params_dict)


    if(args.use_untrained_model):
        config = AutoConfig.from_pretrained(checkpoint_path)
        if args.arch_config_overrides is not None:
            for key, value in json.loads(args.arch_config_overrides).items():
                setattr(config, key, value)
        model = AutoModelForCausalLM.from_config(config)
    else:
        if "gemma" in args.model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, output_scores = True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto" if args.incontext_input else None)
        else:      
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, output_scores = True, return_dict_in_generate=True, device_map="auto" if args.incontext_input else None)


    if args.incontext_input:
        start_time = time.time()
        compute_inference_results(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            selected_token_ids=selected_token_ids,
            incontext_common_prefix_len=incontext_common_prefix_len,
            store_path=output_dir,
            device=next(iter(model.parameters())).device.type,
            batch_size=args.icl_batch_size
        )
        end_time = time.time()

        

    else:

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        training_args = TrainingArguments(
            output_dir = output_dir,
            eval_strategy = args.evaluation_strategy,
            logging_strategy = args.evaluation_strategy,
            logging_steps=args.logging_steps,
            learning_rate = args.learning_rate,
            lr_scheduler_type = args.lr_scheduler,
            warmup_ratio = args.warmup_ratio,
            num_train_epochs = args.num_train_epochs, 
            max_steps=args.max_steps,
            save_strategy = save_strategy,
            eval_accumulation_steps=1,
            save_total_limit=1 if args.save_best_model else None,
            metric_for_best_model="eval_test_sequences_loss" if args.save_best_model else None,
            greater_is_better=False if args.save_best_model else None,
            save_only_model=True,
            # load_best_model_at_end=True,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            auto_find_batch_size=True if not args.use_deepspeed else False,
            run_name = run_name,
            report_to=["wandb"] if local_rank == 0 else ["none"],
            deepspeed=deepspeed_config,
            # gradient_checkpointing = True,
            # gradient_checkpointing_kwargs = {"use_reentrant": False},
            eval_on_start=True
        )



        
        effective_samples_per_batch = int(os.environ["WORLD_SIZE"]) * args.batch_size * args.logging_steps
        if not args.adaptive_training:
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=Subset(dataset["train_sequences"], range(len(dataset["train_sequences"]))),
                eval_dataset=dataset,
                data_collator=data_collator,
                preprocess_logits_for_metrics=preprocess_logits(tokenizer, selected_token_ids),
            )
        else:
            assert args.num_train_epochs == 1 # only one epoch
            assert not args.incontext_input
            assert args.evaluation_strategy == "steps"
            training_args.num_train_epochs = -1
            training_args.max_steps = dataset["train_sequences"].num_rows // (int(os.environ["WORLD_SIZE"]) * args.batch_size)
            
            if local_rank == 0:
                print("Effective Sample size", effective_samples_per_batch)
                print("Effective Steps", training_args.max_steps)
            
            
            trainer = NoShuffleTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=Subset(dataset["train_sequences"].select(range(0, len(dataset["train_sequences"]))), range(len(dataset["train_sequences"]))),
                eval_dataset=dataset,
                data_collator=data_collator,
                preprocess_logits_for_metrics=preprocess_logits(tokenizer, selected_token_ids),
            )

        
        grammar_callback = GrammarCallback(base_config=vars(args),
                                        trainer=trainer, 
                                        tokenizer=tokenizer, 
                                        dataset=dataset, 
                                        incontext_common_prefix_len=incontext_common_prefix_len,
                                        train_test_distance=train_test_distance,
                                        effective_samples_per_batch=effective_samples_per_batch if args.adaptive_training else None
                                        )
        trainer.compute_metrics = compute_metrics(grammar_callback, selected_token_ids)
        generate_text_callback = GenereteTextCallback(tokenizer, 
                                                      dataset, 
                                                      max_new_tokens=args.max_new_tokens, 
                                                      compute_msp=args.compute_msp,
                                                      global_prefix_config=args.global_prefix_config)

        trainer.add_callback(grammar_callback)
        if args.generate_text:
            trainer.add_callback(generate_text_callback)
        


        start_time = time.time()
        trainer.train()
        if args.save_final_checkpoint:
            trainer.save_model(f"{output_dir}/final/") # save the model and tokenizer
        if args.adaptive_training:
            grammar_callback.starting_epoch = trainer.state.epoch
            trainer.eval_dataset = grammar_callback.dataset.copy()

            # sample 4 * args.considered_eval_samples
            trainer.eval_dataset['train_sequences'] = trainer.eval_dataset['train_sequences'].shuffle(seed=42).select(range(min(args.considered_training_samples, 4 * args.considered_eval_samples, len(trainer.eval_dataset['train_sequences']))))

            for eval_dataset in grammar_callback.dataset.keys():
                if eval_dataset != "train_sequences":
                    del trainer.eval_dataset[eval_dataset]

            if local_rank == 0:
                print("End eval...")
                print(trainer.eval_dataset)
            trainer.evaluate()
        end_time = time.time()

    
    if(local_rank == 0):
        wandb.finish()
        # store wandb result locally as pickle
        api = wandb.Api()
        wandb_entity_name = "trustworthy-ml"
        run_id = None
        for file in os.listdir(f"{output_dir}/wandb/latest-run"):
            if file.endswith(".wandb"):
                run_id = file.split(".")[0].split("run-")[-1]
                break
        assert run_id is not None
        runs = api.runs(wandb_entity_name + "/" + wandb_project_name)
        result = {}
        for run in runs:
            if run.id == run_id:
                print(run)
                result['summary'] = run.summary._json_dict
                result['config'] = {k: v for k, v in run.config.items() if not k.startswith("_")}
                result['name'] = run.name
                result['history'] = pd.DataFrame([row for row in run.scan_history()])

                print("Storing file:", f"{output_dir}/run.pkl")
                
                # to pickle
                with open(f"{output_dir}/run.pkl", "wb") as f:
                    pickle.dump(result, f)

                break
    
    if local_rank == 0:
        if args.save_final_checkpoint or args.save_best_model:
            # delete global_step
            for folder in os.listdir(output_dir):
                if os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith('checkpoint'):
                    deleted_folder = f"{output_dir}/{folder}/global_step"
                    os.system(f"rm -rf {deleted_folder}*")

        # store time taken
        with open(f"{output_dir}/time.txt", "w") as f:
            f.write(f"{end_time - start_time}")
        
        # mv everything to NFS
        os.system(f"mkdir -p artifacts")
        os.system(f"mkdir -p {output_dir.replace('/tmp/', 'artifacts/')}")
        os.system(f"mv {output_dir}/* {output_dir.replace('/tmp/', 'artifacts/')}")

        
        # split results for multilingual experiments
        
        if args.multilingual:
            args.multilingual_grammar_name = args.grammar_name
            for i, grammar_name in enumerate(args.grammar_name.split("_aNd_")):
                split_output_dir = f"{output_dir.replace('/tmp/', 'artifacts/').replace(args.multilingual_grammar_name, grammar_name)}_splitted_{i+1}"
                os.system(f"cp -r {output_dir.replace('/tmp/', 'artifacts/')} {split_output_dir}")
                
                # args
                args.grammar_name = grammar_name
                with open(os.path.join(split_output_dir, "args.pkl"), "wb") as f:
                    pickle.dump(vars(args), f)
                with open(os.path.join(split_output_dir, "args.json"), "w") as f:
                    json.dump(vars(args), f)
                
                # results
                for filename in ["grammar_eval_result_average.csv", "grammar_eval_result.csv", "memorization_pruning.csv"]:
                    if not os.path.exists(f"{split_output_dir}/{filename}"):
                        continue
                    df_target = pd.read_csv(f"{split_output_dir}/{filename}")
                    relevant_eval_datasets = [
                        eval_dataset for eval_dataset in df_target["eval_dataset"].unique() if eval_dataset.endswith(f"_g{i+1}")
                    ]
                    eval_dataset_map = {
                        eval_dataset: eval_dataset.replace(f"_g{i+1}", "")
                        for eval_dataset in relevant_eval_datasets
                    }
                    df_target = df_target[df_target["eval_dataset"].isin(relevant_eval_datasets)].copy()
                    df_target["eval_dataset"] = df_target["eval_dataset"].apply(lambda x: eval_dataset_map[x])
                    df_target.to_csv(f"{split_output_dir}/{filename}", index=False)
                

            


if __name__ == "__main__":
    args = get_args()
    args.learning_rate = lr_dict[args.model_name] if args.model_name in lr_dict else args.learning_rate
    if not args.incontext_input:
        args.batch_size = batch_size_dict[args.model_name] if args.model_name in batch_size_dict else args.batch_size
        
    dataset_dict, max_sequence_length, unique_tokens = get_data(args)

    # preprocessing for memorization intervention
    distance_based_result = None
    if "remove" in args.memorization_algo:
        meta_data_filename = f"../data/{args.grammar_name}/meta_data_{args.grammar_name}_10000_5.pkl"
        if os.path.exists(meta_data_filename):
            with open(meta_data_filename, 'rb') as f:
                string_meta_data = pickle.load(f)
                distance_based_result = min_distant_sequences(dataset_dict['train_sequences']['text'], dataset_dict['test_sequences']['text'], string_meta_data['sequence_prob_dict'])
    
    training(args, dataset_dict, max_sequence_length, unique_tokens, distance_based_result)