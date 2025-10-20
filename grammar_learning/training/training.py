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
    create_dataset_dict, 
    get_args, 
    get_parser,
    get_selected_token_ids,
    GenereteTextCallback,
    GrammarCallback,
    compute_metrics,
    preprocess_logits,
    compute_inference_results,
    process_for_under_trained_tokens,
    text_generation,
    min_distant_sequences
)
import json
import pandas as pd
from copy import deepcopy
import wandb
import logging
import time
from datetime import datetime
import accelerate
from torch.utils.data import Subset




def training(args, dataset_dict, max_sequence_length, unique_tokens, train_test_distance):

    
    logger = logging.getLogger(__name__)
    set_seed(args.run_seed)
    tokenizer, checkpoint_path = get_tokenizer(args)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id

    selected_token_ids = get_selected_token_ids(tokenizer, unique_tokens, logger)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except:
        assert args.incontext_input
        local_rank = 0

    # selected models, where cpu offloading is needed
    model_offloading = ["mistralai/Mistral-7B-v0.3", 
                        "mistralai/Mistral-Nemo-Base-2407", 
                        "meta-llama/Meta-Llama-3-8B", 
                        "meta-llama/Meta-Llama-3.1-8B", 
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-13b-hf",
                        "base_models_vnanda/Llama-2-13b-hf",
                        "google/gemma-2-9b",
                        "Qwen/Qwen2.5-14B"
    ]
    deepspeed_config = "additional/deepspeed_config.json" if args.use_deepspeed and args.model_name not in model_offloading else (
                  "additional/deepspeed_config_offloading.json" if args.use_deepspeed else None
    )        
    
    
    # tokenize
    if not args.nlp_dataset:
        # apply character level tokenization
        encoded_dataset  = encode_dataset(tokenizer=tokenizer, 
                                        dataset=dataset_dict, 
                                        max_sequence_length=max_sequence_length, 
                                        logger=logger,
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
        
    # unused
    if args.mem_no_batch:
        # batch size is training size
        args.batch_size = dataset['train_sequences'].num_rows
        print("Changing batch size to", args.batch_size)
        deepspeed_config = "additional/deepspeed_config_offloading.json"
        print("Applying CPU overloading")
        
    

    current_time = datetime.now()

    evaluation_strategy = 'epoch'    # evaluation_strategy = 'steps'
    logging_strategy = evaluation_strategy
    save_strategy = evaluation_strategy if args.save_checkpoint or args.save_best_model else 'no'
    if args.incontext_input:
        save_strategy = 'no'

    # output directory (initially stored in /tmp, and later moved to ./artifacts/)
    incontext_common_prefix_len = None
    if args.incontext_input:
        output_dir = f"/tmp/inference_{current_time.strftime('%Y_%m_%d_%H_%M')}_{args.model_name.replace('/', '_')}_{args.grammar_name}_{args.num_samples}_{args.run_seed}_{args.comment.replace(' ', '_')}"
        if args.incontext_input: # only for inference
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

    lr_scheduler = args.lr_scheduler
    warmup_ratio = args.warmup_ratio
    run_name = f"gl | {args.model_name} | {args.grammar_name} | {args.num_samples} | {current_time.strftime('%Y_%m_%d_%H_%M')}"

    # for wandb
    params_dict = {
        'max_sequence_length': max_sequence_length,
        'lr_scheduler': lr_scheduler,
        'warmup_ratio' : warmup_ratio,
        'output_dir': output_dir,
    }

    for key, value in vars(args).items():
        params_dict[key] = value

    if(local_rank == 0):
        os.environ["WANDB_WATCH"] = "all"
        os.environ["WANDB_API_KEY"]="cd8d79cbe96f9d1e1d5fbf1b4829ee38e3e2f76f"
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
            eval_strategy = evaluation_strategy,
            logging_strategy = logging_strategy,
            # logging_steps=args.logging_steps,
            learning_rate = args.learning_rate,
            lr_scheduler_type = lr_scheduler,
            warmup_ratio = warmup_ratio,
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



        
        

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=Subset(dataset["train_sequences"], range(len(dataset["train_sequences"]))),
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
                                        )
        trainer.compute_metrics = compute_metrics(grammar_callback, selected_token_ids)
        generate_text_callback = GenereteTextCallback(tokenizer, 
                                                      dataset, 
                                                      max_new_tokens=args.max_new_tokens, 
                                                      compute_msp=args.compute_msp,
                                                      global_prefix_config=args.global_prefix_config)

        if args.store_result:
            trainer.add_callback(grammar_callback)
        if args.generate_text:
            trainer.add_callback(generate_text_callback)
        


        start_time = time.time() 
        if not args.incontext_input:
            trainer.train()
            if args.save_final_checkpoint:
                trainer.save_model() # save the model and tokenizer
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

        


if __name__ == "__main__":
    args = get_args(get_parser())
    lr_dict = {
        "EleutherAI/pythia-6.9b": 0.00001,
        "EleutherAI/pythia-1b": 0.00001,
        "EleutherAI/pythia-2.8b": 0.00001,
        
        "mistralai/Mistral-7B-v0.3": 0.000005,
        "mistralai/Mistral-Nemo-Base-2407": 0.000005,
        
        "meta-llama/Meta-Llama-3-8B": 0.00005,
        "meta-llama/Meta-Llama-3.1-8B": 0.00005,
        "meta-llama/Llama-3.2-1B": 0.00005,
        "meta-llama/Llama-3.2-3B": 0.00005,

        "google/gemma-2-2b": 0.00005,
        "google/gemma-2-9b": 0.00005,

        "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf": 0.000005,
        "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf": 0.000005,

        "/NS/llm-1/nobackup/soumi/opt-model-1.3B": 0.000005,
        "/NS/llm-1/nobackup/soumi/opt-model-2.7B": 0.000005,
        "/NS/llm-1/nobackup/soumi/opt-model-6.7B": 0.000005,

        "Qwen/Qwen2.5-0.5B":0.00005,
        "Qwen/Qwen2.5-1.5B":0.00005,
        "Qwen/Qwen2.5-7B":0.00005,
        "Qwen/Qwen2.5-14B":0.00005,
    }

    batch_size_dict = {
        "EleutherAI/pythia-6.9b": 8,
        "EleutherAI/pythia-1b": 8,
        "EleutherAI/pythia-2.8b": 8,
        
        "mistralai/Mistral-7B-v0.3": 8,
        "mistralai/Mistral-Nemo-Base-2407": 8,
        
        "meta-llama/Meta-Llama-3-8B": 8,
        "meta-llama/Meta-Llama-3.1-8B": 8,
        "meta-llama/Llama-3.2-1B": 8,
        "meta-llama/Llama-3.2-3B": 8,

        "google/gemma-2-2b": 8,
        "google/gemma-2-9b": 8,

        "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf": 8,
        "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf": 8,


        "/NS/llm-1/nobackup/soumi/opt-model-1.3B": 8,
        "/NS/llm-1/nobackup/soumi/opt-model-2.7B": 8,
        "/NS/llm-1/nobackup/soumi/opt-model-6.7B": 8,

        "Qwen/Qwen2.5-14B": 4,
    }

    
    args.learning_rate = lr_dict[args.model_name] if args.model_name in lr_dict else args.learning_rate
    if not args.incontext_input:
        args.batch_size = batch_size_dict[args.model_name] if args.model_name in batch_size_dict else args.batch_size
        
    data_dict, max_sequence_length, unique_tokens = get_data(args)

    # preprocessing for memorization intervention
    distance_based_result = None
    if "remove" in args.memorization_algo:
        meta_data_filename = f"../data/{args.grammar_name}/meta_data_{args.grammar_name}_10000_5.pkl"
        if os.path.exists(meta_data_filename):
            with open(meta_data_filename, 'rb') as f:
                string_meta_data = pickle.load(f)
                distance_based_result = min_distant_sequences(data_dict['train_sequences'], data_dict['test_sequences'], string_meta_data['sequence_prob_dict'])
    dataset_dict = create_dataset_dict(data_dict)
    training(args, dataset_dict, max_sequence_length, unique_tokens, distance_based_result)