import torch
import transformers
import os
import numpy as np
import logging
# from transformers import GemmaTokenizerFast
from datasets import Dataset, DatasetDict
import pickle
from transformers import AutoTokenizer
from transformers import AutoConfig
import argparse
import random
import pandas as pd
from transformers import TrainerCallback
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
import torch.distributed as dist
import time
from copy import deepcopy

    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--checkpoint_path_overwrite", type=str, default=None, help="Checkpoint path overwrite")
    parser.add_argument("--use_untrained_model", action='store_true', help="Use untrained model")
    parser.add_argument("--grammar_name", type=str, default="anbn", help="Grammar name")
    parser.add_argument("--data_comment", type=str, default=None, help="Data comment")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of sequences")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--comment", type=str, default="", help="Comment")
    parser.add_argument("--store_result", action='store_true', help="Store result")
    parser.add_argument("--generate_text", action='store_true', help="Store result")
    parser.add_argument("--considered_training_samples", type=int, default=None, help="Considered training samples")
    parser.add_argument("--skip_training_samples", type=int, default=0, help="Skip training samples")
    parser.add_argument("--considered_eval_samples", type=int, default=128, help="Considered training samples")
    parser.add_argument("--considered_incontext_examples", type=int, default=0, help="Considered incontext samples")
    parser.add_argument("--considered_incontext_repetitions", type=int, default=1, help="How many times to repeat in-context experiments")
    parser.add_argument("--incontext_data_source", type=str, default=None, help="Source pickle file and dataset name separated by colon")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=-1, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for training")
    parser.add_argument("--data_seed", type=int, default=0, help="Random seed for data")
    parser.add_argument("--run_seed", type=int, default=None, help="Random seed for experiments")
    parser.add_argument("--exclude_test_data", action='store_true', help="Exclude test data")
    parser.add_argument("--include_edit_distance_eval", action='store_true', help="Include edit distance eval datasets")
    parser.add_argument("--include_grammar_edit_eval", action='store_true', help="Include grammar_edit eval datasets")
    parser.add_argument("--include_incorrect_random_eval", action='store_true', help="Include incorrect random eval datasets")
    parser.add_argument("--combine_edit_distance", action='store_true', help="Combine edit distance datasets")
    parser.add_argument("--save_checkpoint", action='store_true', help="Save checkpoint")
    parser.add_argument("--save_final_checkpoint", action='store_true', help="Save final checkpoint")
    parser.add_argument("--save_best_model", action='store_true', help="Save the best checkpoint")
    parser.add_argument("--inference_only_mode", action='store_true', help="Inference only mode")
    parser.add_argument("--incontext_input", action='store_true', help="Process input as incontext input")
    parser.add_argument("--use_deepspeed", action='store_true', help="Use deepspeed")
    parser.add_argument("--max_new_tokens", default=1, type=int, help="Max new tokens to generate in text generation mode")
    parser.add_argument("--compute_msp", action='store_true', help="Compute minimum sufficient prefix")
    parser.add_argument("--incontext_separator", type=str, default="semicolon", help="Separator in in-context learning experiemnts")
    parser.add_argument("--run_in_docker", action='store_true', help="Run in docker")
    parser.add_argument("--lr_scheduler", type=str, default="linear", help="Learning schedule", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    

    # counterfactural memorization
    parser.add_argument("--counterfactual_memorization", action='store_true', help="Counterfactual memorization")
    parser.add_argument("--counterfactual_string_index", type=int, default=0, help="Counterfactual string index")
    parser.add_argument("--mem_no_batch", action='store_true', help="Whether to put all training strings in one batch or not")

    parser.add_argument("--use_under_trained_tokens", action='store_true', help="Use untrained tokens")
    parser.add_argument("--icl_batch_size", type=int, default=8, help="Batch size for ICL")

    # local prefix
    parser.add_argument("--global_prefix_config", type=str, default='no_global_prefix', help="Configuration of global prefix")

    # memorization intervention
    parser.add_argument("--memorization_intervention", type=str, default=None, help="Memorization intervention pivot directory")
    parser.add_argument("--memorization_approach", type=str, default="contextual_memorization", help="Memorization intervention approach")
    return parser



separator_dict = {
    "space": " ",
    "semicolon": ";",
    "comma": ",",
    "colon": ":",
    "period": "."
}



def process_for_under_trained_tokens(args, tokenizer, dataset, selected_token_ids):
    under_trained_token_id_list = {
        "mistralai/Mistral-7B-v0.3": [32506, 21186, 27404, 27175, 27160, 26851, 19527, 10591, 26601, 8376, 28939, 23907, 15824, 18463, 32131, 12961, 17711, 15524, 21460, 11046],
        "EleutherAI/pythia-6.9b": [26868, 28696, 17030, 37402, 41606, 26362, 15479, 30356, 14798, 39743, 15236],
        "Qwen/Qwen2.5-7B": [78783, 79269, 79270, 83969, 83971, 142386, 97000, 136954, 78323, 88372, 142494, 88371, 138175, 122290, 122474, 127734, 151293, 122223, 122578, 117332],
        "meta-llama/Meta-Llama-3.1-8B": [85071, 107658, 127896, 103003, 126523, 80369, 79883, 106710, 68896, 118508, 89472, 127117, 126647, 124292, 122549, 122746, 64424, 85069, 80370, 125952]
    }

    if args.model_name not in under_trained_token_id_list:
        raise Exception(f"Model {args.model_name} not supported for under trained tokens")

    # mapping
    under_trained_token_id_map = {}
    idx = 0
    for token_id in selected_token_ids:
        if idx >= len(under_trained_token_id_list[args.model_name]) or tokenizer.encode(separator_dict[args.incontext_separator])[0] == token_id:
            under_trained_token_id_map[token_id] = token_id
        else:
            under_trained_token_id_map[token_id] = under_trained_token_id_list[args.model_name][idx]
            idx += 1
    print(under_trained_token_id_map)

    def apply_token_id_map(dataset, token_id_map):
        for token_id in token_id_map:
            dataset["input_ids"][dataset["input_ids"] == token_id] = token_id_map[token_id]
            
        return dataset
    
    dataset = dataset.map(apply_token_id_map, fn_kwargs={"token_id_map": under_trained_token_id_map})

    return dataset, selected_token_ids

def get_args(parser):
    args = parser.parse_args()
    assert args.max_steps != -1 or args.num_train_epochs != -1, "Either max_steps or num_train_epochs should be specified"

    if args.incontext_input:
        assert args.inference_only_mode
    
    if args.inference_only_mode:
        args.use_deepspeed = False

    return args

def prepare_input_for_incontext(data_dict, num_incontext_examples, num_incontext_repetitions=1, incontext_data_source=None, separator="semicolon", seed=5):   
    separator = separator_dict[separator]

    # From training examples
    incontext_common_prefix = []
    incontext_dataset = None
    if incontext_data_source is not None:
        [incontext_data_source_filename, incontext_dataset_name] = incontext_data_source.split(":")
        incontext_dataset = pickle.load(open(incontext_data_source_filename, "rb"))[incontext_dataset_name]
        random.seed(seed)
        random.shuffle(incontext_dataset)
        print(f"Applying incontext learning from {incontext_data_source_filename} with dataset {incontext_dataset_name}")
    else:
        assert "train_sequences" in data_dict.keys()
        incontext_dataset = data_dict["train_sequences"]
        print(f"Applying incontext learning from training data")

    
    for _ in range(num_incontext_repetitions):
        for sequence in incontext_dataset[:num_incontext_examples]:
            incontext_common_prefix.extend(list(sequence))
            incontext_common_prefix.append(separator)


    result = {}
    for key in data_dict.keys():
        # if key == "train_sequences":
        #     # result[key] = data_dict[key]
        #     # continue
        #     data_dict[key] = data_dict[key][:max(1, num_incontext_examples)]
        
        # result[key] = []
        # for sequence in data_dict[key]:
        #     result[key].append(tuple(incontext_common_prefix + list(sequence)))
    
        result[key] = data_dict[key]
    if len(incontext_common_prefix) > 0:
        result['incontext_common_prefix'] = [tuple(incontext_common_prefix)]

    return result, incontext_common_prefix

def get_data(args, verbose=False):
    data_path = "../data"
    if("data_comment" in vars(args) and args.data_comment is not None):
        filename = f"{data_path}/{args.grammar_name}/sequences_w_edit_distance_{args.grammar_name}_{args.num_samples}_{args.data_seed}_{args.data_comment}.pkl"
    else:
        filename = f"{data_path}/{args.grammar_name}/sequences_w_edit_distance_{args.grammar_name}_{args.num_samples}_{args.data_seed}.pkl"
    
    if os.path.exists(filename):
        print(f"Loading sequences from {filename}")
        with open(filename, 'rb') as f:
            raw_data_dict = pickle.load(f)
            assert 'train_sequences' in raw_data_dict.keys()
            if args.run_seed is None:
                args.run_seed = args.data_seed
                # no need to shuffle
            else:
                random.seed(args.run_seed)
                random.shuffle(raw_data_dict['train_sequences'])
            if args.grammar_name == "pcfg_g1_g2_combined":
                print("Merging train sequences of two grammars")
                train_sequences = []
                train_sequences_g1 = []
                train_sequences_g2 = []
                for sequence in raw_data_dict['train_sequences']:
                    assert len(sequence) == 2
                    train_sequences.append(sequence[0])
                    train_sequences.append(sequence[1])

                    train_sequences_g1.append(sequence[0])
                    train_sequences_g2.append(sequence[1])
                    
                raw_data_dict['train_sequences'] = train_sequences
                raw_data_dict['train_sequences_g1'] = train_sequences_g1
                raw_data_dict['train_sequences_g2'] = train_sequences_g2

                # a test set combining two grammars
                raw_data_dict['test_sequences'] = raw_data_dict['test_sequences_g1'][:args.considered_eval_samples//2] + raw_data_dict['test_sequences_g2'][:args.considered_eval_samples//2]

                raw_data_dict['train_sequences_g1'] = raw_data_dict['train_sequences_g1'][:args.considered_eval_samples]
                raw_data_dict['train_sequences_g2'] = raw_data_dict['train_sequences_g2'][:args.considered_eval_samples]



            raw_data_dict['train_sequences'] = raw_data_dict['train_sequences'][args.skip_training_samples:]
            # assert 'test_sequences' in raw_data_dict.keys()
            # if "test_sequences" in raw_data_dict.keys():
            #     assert len(raw_data_dict['train_sequences']) + len(raw_data_dict['test_sequences']) == args.num_samples

    else:
        raise ValueError(f"File {filename} does not exist")

    if(args.considered_training_samples is not None):
        assert args.considered_training_samples >= 0
        if args.considered_training_samples == 0 and args.incontext_input:
            args.considered_training_samples = 1
        raw_data_dict['train_sequences'] = raw_data_dict['train_sequences'][:args.considered_training_samples]

    # combine edit distance diff position into 1 datasetx
    if('combine_edit_distance' in vars(args) and args.combine_edit_distance):
        modified_data_dict = {}
        delete_keys = []
        for key in raw_data_dict.keys():
            if("edit_distance" in key):
                split = key.split("_")
                new_key = "_".join(split[:-2])
                if(new_key not in modified_data_dict):
                    modified_data_dict[new_key] = raw_data_dict[key]
                else:
                    modified_data_dict[new_key] += raw_data_dict[key]
                delete_keys.append(key)
        # random sample len(test_sequences) data
        for key in modified_data_dict.keys():
            # shuffle
            random.seed(args.data_seed)
            random.shuffle(modified_data_dict[key])
            if 'test_sequences' in raw_data_dict.keys():
                modified_data_dict[key] = modified_data_dict[key][:len(raw_data_dict['test_sequences'])]
            else:
                modified_data_dict[key] = modified_data_dict[key][:args.considered_eval_samples]
        raw_data_dict.update(modified_data_dict)
        for key in delete_keys:
            del raw_data_dict[key]

    # counterfactual memorization
    if args.counterfactual_memorization:
        print("Counterfactual memorization")
        assert f"counterfactual_{args.counterfactual_string_index}" in raw_data_dict
        # print(len(raw_data_dict['train_sequences']))
        considered_counterfactual_string = max(int((args.considered_training_samples * len(raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"] * 2))/args.num_samples), 1)
        print(f"Initial considered_counterfactual_string:", len(raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"]))
        print(f"considered_counterfactual_string: {considered_counterfactual_string}") 
        for counterfactual_string in raw_data_dict[f"counterfactual_{args.counterfactual_string_index}"][:considered_counterfactual_string]:
            raw_data_dict['train_sequences'].append(counterfactual_string)

        # quit()
        # print(len(raw_data_dict['train_sequences']))
        # shuffle
        random.shuffle(raw_data_dict['train_sequences'])
        # for sequence in raw_data_dict['train_sequences'][-10:]:
        #     print(sequence[:20])
        # quit()
        


    if args.incontext_input:
        raw_data_dict, incontext_common_prefix = prepare_input_for_incontext(raw_data_dict, 
                                                                             num_incontext_examples=args.considered_incontext_examples,
                                                                             num_incontext_repetitions=args.considered_incontext_repetitions, 
                                                                             incontext_data_source=args.incontext_data_source,
                                                                             separator=args.incontext_separator,
                                                                             seed=args.run_seed
        )

        # training dataset can be a bottleneck
        # print("-----------------Shuffling and restricting training dataset (ICL only)-----------------")
        # random.shuffle(raw_data_dict['train_sequences'])
        # raw_data_dict["train_sequences"] = raw_data_dict["train_sequences"][:args.considered_eval_samples]

        model_config = AutoConfig.from_pretrained(args.model_name).to_dict()
        max_position_embeddings = model_config['max_position_embeddings'] if 'max_position_embeddings' in model_config else(
                        model_config['n_positions'] if 'n_positions' in model_config else None
        )
        if len(incontext_common_prefix) > max_position_embeddings:
            print("Error! Incontext input is too long!")
            quit()
        args.considered_incontext_examples = (args.considered_incontext_examples, incontext_common_prefix)
    
    max_seq_len_list = []
    min_seq_len_list = []
    unique_tokens = {}
    data_dict = {}
    for key in raw_data_dict.keys():
        # filter eval_datasets
        if(not args.include_edit_distance_eval and ("edit_distance" in key or "grammar_edit" in key)):
            continue
        if(not args.include_grammar_edit_eval and "grammar_edit" in key):
            continue
        if(not args.include_incorrect_random_eval and "non_grammatical" in key):
            continue
        if args.exclude_test_data and "test" in key:
            continue

        if "edit" in key and "train" in key:
            # edits on the training data are not considered at all
            continue
        
        if "train_sequences" in key:
            data_dict[key] = raw_data_dict[key]
        else:
            data_dict[key] = raw_data_dict[key][:args.considered_eval_samples]
        
        if len(data_dict[key]) > 0 and key != "incontext_common_prefix":
            max_seq_len_list.append(max(len(s) for s in data_dict[key]))
            min_seq_len_list.append(min(len(s) for s in data_dict[key]))
        
        for sentence in data_dict[key]:
            for token in sentence:
                if token not in unique_tokens:
                    unique_tokens[token] = 1
                else:
                    unique_tokens[token] += 1
    max_sequence_length = max(max_seq_len_list)
    
    if(verbose):
        for key in data_dict.keys():
            print(key)
            print(data_dict[key][:5])
            print()

        print(unique_tokens)
        # quit()

        print(max_seq_len_list, min_seq_len_list, max_sequence_length)
    return data_dict, max_sequence_length, list(unique_tokens.keys())



# helper code
def create_dataset_dict(data_dict):
    # torch compatible
    dataset_dict = {}
    for key in data_dict.keys():
        # if("edit_distance_2" in key):
        dataset_dict[key] = Dataset.from_dict({"text": data_dict[key]})

    datasets = DatasetDict(dataset_dict)
    datasets.set_format(type="torch", columns=["text"])
    return datasets

def get_tokenizer(args, load_tokenizer=True, use_local_path=False):
    # if load_tokenizer is false, it means we are interested in only knowing the checkpoint_path, which should be the original model path not the fine-tuned one
    if args.checkpoint_path_overwrite is None or not load_tokenizer:
        if use_local_path:
            raise ValueError()
        else:
            checkpoint_path = args.model_name
    else:
        # assert args.model_name in args.checkpoint_path_overwrite
        checkpoint_path = args.checkpoint_path_overwrite

    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, checkpoint_path
    else:
        return checkpoint_path
    



def set_path(save_root='result', save_tag=""):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    if(save_tag == ""):
        save_tag = f"llm_graph_of_toughts"
    exp_seq_path = os.path.join(save_root, 'exp_seq.txt')

    if not os.path.exists(exp_seq_path):
        file = open(exp_seq_path, 'w')
        exp_seq=0
        exp_seq = str(exp_seq)
        file.write(exp_seq)
        file.close
        save_tag = 'exp_' + exp_seq + '_' + save_tag
    else:
        file = open(exp_seq_path, 'r')
        exp_seq = int(file.read())
        exp_seq += 1
        exp_seq = str(exp_seq)
        save_tag = 'exp_' + exp_seq + '_' + save_tag
        file = open(exp_seq_path, 'w')
        file.write(exp_seq)
        file.close()

    exp_seq = exp_seq
    save_path = os.path.join(save_root, save_tag)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config_path = os.path.join(save_path, 'config.json')
    logger_path = os.path.join(save_path, 'exp_log.log')    


    return logger_path, exp_seq, save_path


def get_logger(save_root='result', save_tag=""):

    logger_path, exp_seq, save_path = set_path(save_root=save_root, save_tag=save_tag)

    logging.basicConfig(
        filename=logger_path,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger, exp_seq, save_path



def tokenize(tokenizer, text, max_length, logger):
    if tokenizer.padding_side == "right":
        if logger is not None:
            logger.warning("Padding side is right, setting it to left")
        tokenizer.padding_side = "left"
    if max_length is None:
        padding = "longest"
    else:
        padding = "max_length"
    return tokenizer(
        text,
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        padding=padding,
        max_length=max_length,
    )

def characterwise_encoding(tokenizer, dataset, max_length, logger, verbose=False):
    sequences = dataset["text"]
    # max_length = max(len(s) for s in sequences)
    sequence_token_ids = []
    sequence_token_masks = []
    for sequence in sequences:
        sequence_chars = list(sequence)
        
        encoded_chars = tokenize(
            tokenizer,
            sequence_chars,
            max_length=4,            
            logger=logger
        )
        
        
        # add end of sentence token
        num_padding = max_length - len(sequence) + 1 # +1 for end of sentence token
        padded_input_ids = torch.cat(
            (
                torch.tensor([tokenizer.pad_token_id] * num_padding, dtype=torch.long),
                torch.tensor([tokenizer.bos_token_id] * 1, dtype=torch.long),
                encoded_chars.input_ids[:, -1:].squeeze(1),
                torch.tensor([tokenizer.eos_token_id] * 1, dtype=torch.long),
            )
        )
        padded_attention_mask = torch.cat(
            (
                torch.tensor([0] * num_padding, dtype=torch.long),
                torch.tensor([0] * 1, dtype=torch.long),
                encoded_chars.attention_mask[:, -1:].squeeze(1),
                torch.tensor([1] * 1, dtype=torch.long),
            )
        )
        # if(isinstance(tokenizer, GemmaTokenizerFast)):
        #     # BOS token at the beginning
        #     padded_input_ids = torch.cat(
        #         (
        #             torch.tensor([tokenizer.pad_token_id] * num_padding, dtype=torch.long),
        #             torch.tensor([tokenizer.bos_token_id] * 1, dtype=torch.long),
        #             encoded_chars.input_ids[:, 1:].squeeze(1),
        #             torch.tensor([tokenizer.eos_token_id] * 1, dtype=torch.long),
        #         )
        #     )
        #     padded_attention_mask = torch.cat(
        #         (
        #             torch.tensor([0] * num_padding, dtype=torch.long),
        #             torch.tensor([0] * 1, dtype=torch.long),
        #             encoded_chars.attention_mask[:, 1:].squeeze(1),
        #             torch.tensor([1] * 1, dtype=torch.long),
        #         )
        #     )
        # else:
        #     padded_input_ids = torch.cat(
        #         (
        #             torch.tensor([tokenizer.pad_token_id] * num_padding, dtype=torch.long),
        #             encoded_chars.input_ids.squeeze(1),
        #             torch.tensor([tokenizer.eos_token_id] * 1, dtype=torch.long),
        #         )
        #     )
        #     padded_attention_mask = torch.cat(
        #         (
        #             torch.tensor([0] * num_padding, dtype=torch.long),
        #             encoded_chars.attention_mask.squeeze(1),
        #             torch.tensor([1] * 1, dtype=torch.long),
        #         )
        #     )
    

        sequence_token_ids.append(padded_input_ids)
        sequence_token_masks.append(padded_attention_mask)
    if(verbose):
        print({
            "input_ids": torch.stack(sequence_token_ids),
            "attention_mask": torch.stack(sequence_token_masks),
        })
        print(torch.stack(sequence_token_ids).shape)
    return {
        "input_ids": torch.stack(sequence_token_ids),
        "attention_mask": torch.stack(sequence_token_masks),
    }



# def encode_dataset(tokenizer, dataset, batch_size, max_sequence_length, logger, verbose=True):
def encode_dataset(tokenizer, dataset, max_sequence_length, logger, verbose=True):
    return dataset.map(
        lambda dataset_split: characterwise_encoding(tokenizer, dataset_split, max_sequence_length, logger, verbose),
        batched=True,
        batch_size=128,
    )
    


def custom_tokenize_string(tokens, attention, position):
    custom_input = transformers.tokenization_utils_base.BatchEncoding()
    custom_input['input_ids'] = []
    custom_input['attention_mask'] = []
    for tvalue, avalue in zip(tokens[:position], attention[:position]):
        custom_input['input_ids'].append(tvalue)
        custom_input['attention_mask'].append(avalue)
    custom_input['input_ids'] = torch.tensor(custom_input['input_ids']).unsqueeze(0)
    custom_input['attention_mask'] = torch.tensor(custom_input['attention_mask']).unsqueeze(0)
    return custom_input



def custom_tokenize_string_batch(tokens, attention, pad_token_id, max_position=None):
    custom_input = transformers.tokenization_utils_base.BatchEncoding()
    custom_input['input_ids'] = []
    custom_input['attention_mask'] = []
    if max_position is None:
        max_position = len(tokens)
    assert max_position >= 1

    for i in range(1, max_position):
        num_padding = max_position - i
        custom_input['input_ids'].append([pad_token_id] * num_padding + list(tokens[:i]))
        custom_input['attention_mask'].append([0] * num_padding + list(attention[:i]))

    custom_input['input_ids'] = torch.tensor(custom_input['input_ids'])
    custom_input['attention_mask'] = torch.tensor(custom_input['attention_mask'])
    return custom_input


def get_selected_token_ids(tokenizer, unique_tokens, logger):
    selected_token_ids = tokenize(
        tokenizer,
        unique_tokens,
        max_length=4,            
        logger=logger
    ).input_ids[:, -1:].squeeze(1).tolist()
    return selected_token_ids
    





class GenereteTextCallback(TrainerCallback):

    def __init__(self, 
                tokenizer, 
                dataset, 
                max_new_tokens, 
                compute_msp, 
                local_prefix_length_list = [5, 10, 20],
                skip_tokens=20,
                generation_interval=1,
                selective_samples=True,
                global_prefix_config = 'random_tokens'):


        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.compute_msp = compute_msp
        self.local_prefix_length_list = local_prefix_length_list
        self.skip_tokens = skip_tokens
        self.generation_interval = generation_interval
        self.selective_samples = selective_samples
        self.global_prefix_config = global_prefix_config
        assert self.global_prefix_config in ['random_token', 'same_language', 'no_global_prefix']

    def remove_eos(self, token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        length = len(token_ids_raw)
        for i, (t, a) in enumerate(zip(token_ids_raw, attentions_raw)):
            if(t == self.tokenizer.eos_token_id and i < length - 1):
                continue
            token_ids.append(t)
            attentions.append(a)
        return token_ids, attentions

    def on_step_end(self, args, state, control, **kwargs):
        if args.local_rank != 0:
            return
        
        if self.compute_msp:
            if state.epoch % 1 != 0 and state.epoch != 1:
                return
            
            
        else:
            if state.epoch % 1 != 0 and state.epoch != 1:
                return
        
        EPS = 1e-12
        
        print("Epoch:", state.epoch)
        
        model = kwargs['model']

        for eval_dataset in self.dataset:
            # ground_truths = []
            ground_truth_token_ids_all = []
            # prompts = []
            prompt_token_ids_all = []
            example_ids = []
            # generated_texts = []
            generated_token_ids_all = []
            length_token_ids_all = []


            random_index_list = None
            if self.compute_msp:
                if eval_dataset != "train_sequences":
                    continue


                msp_prefix_length = []
                original_prompt_token_ids = []
                prompt_ids = []
                random_index = []
                generated_token_negative_log_prob_all = []

                if not self.selective_samples:
                    max_index = 20
                    np.random.seed(0)
                    if self.dataset[eval_dataset].shape[0] > max_index:
                        random_index_list = np.random.choice(
                            self.dataset[eval_dataset].shape[0], size=max_index, replace=False
                        )
                    else:
                        random_index_list = np.arange(self.dataset[eval_dataset].shape[0])

                else:
                    sequence_to_index_map = {}
                    for index, token_id in enumerate(self.dataset[eval_dataset]['input_ids']):
                        token_id = tuple(token_id.cpu().numpy())
                        if token_id not in sequence_to_index_map:
                            sequence_to_index_map[token_id] = []
                        sequence_to_index_map[token_id].append(index)

                
                    sequence_freq = {}
                    for sequence in sequence_to_index_map:
                        sequence_freq[sequence] = len(sequence_to_index_map[sequence])
                    sequence_freq = dict(sorted(sequence_freq.items(), key=lambda x: x[1], reverse=True))

                    
                    # take max, median, and min
                    max_idx = 0
                    min_idx = -1
                    median_idx = len(sequence_freq) // 2
                    random_index_list = [sequence_to_index_map[list(sequence_freq.keys())[max_idx]][0],
                                        sequence_to_index_map[list(sequence_freq.keys())[median_idx]][0],
                                        sequence_to_index_map[list(sequence_freq.keys())[min_idx]][0]]
                    
                print(f"Random index list: {random_index_list}")


            dataset_token_ids = []
            for index in tqdm(range(len(self.dataset[eval_dataset]))):
                token_ids_raw, attention_raw = self.dataset[eval_dataset]['input_ids'].tolist()[index], self.dataset[eval_dataset]['attention_mask'].tolist()[index]
                token_ids, attention = self.remove_eos(token_ids_raw, attention_raw) # this turns out to be a good idea
                token_ids = np.array(token_ids)
                dataset_token_ids.append(token_ids)

            
            for index in tqdm(range(len(self.dataset[eval_dataset]))):
                if self.compute_msp and index not in random_index_list:
                    continue

                token_ids_raw, attention_raw = self.dataset[eval_dataset]['input_ids'].tolist()[index], self.dataset[eval_dataset]['attention_mask'].tolist()[index]
                token_ids, attention = self.remove_eos(token_ids_raw, attention_raw) # this turns out to be a good idea
                token_ids = np.array(token_ids)
                                

                prompt_token_ids = []
                # model_responses = []
                token_length = token_ids.shape[0]    
                for i in range(1, token_ids.shape[0] - 1):
                    
                    if i % self.generation_interval != 0 or i + self.max_new_tokens > token_length or i <= self.skip_tokens:
                        continue
                    

                    if self.compute_msp:
                        assert self.max_new_tokens == 1
                        for prefix_length in self.local_prefix_length_list + [i]:
                            if prefix_length > i:
                                continue
                            # print("Prefix length:", prefix_length)
                        
                            for rand_idx in range(5):
                                
                                if self.global_prefix_config == 'same_language':
                                    dataset_token_ids_sufficient = []
                                    for token_ids_temp in dataset_token_ids:
                                        if len(token_ids_temp) >= i - prefix_length:
                                            dataset_token_ids_sufficient.append(token_ids_temp)

                                    if len(dataset_token_ids_sufficient) == 0:
                                        continue
                                    

                                    random_remote_prefix_full = dataset_token_ids_sufficient[np.random.choice(len(dataset_token_ids_sufficient))]
                                    random_remote_prefix = random_remote_prefix_full[:i-prefix_length].copy()

                                elif self.global_prefix_config == 'random_token':
                                    random_remote_prefix = token_ids[:i-prefix_length].copy()
                                    np.random.seed(rand_idx)
                                    np.random.shuffle(random_remote_prefix)

                                elif self.global_prefix_config == 'no_global_prefix':
                                    if rand_idx > 0:
                                        continue
                                    random_remote_prefix = token_ids[i-prefix_length:i-prefix_length].copy()
                                    
                                else:
                                    raise ValueError(self.global_prefix_config)
                            

                                
                                local_token_ids = token_ids[i-prefix_length:i]
                                token_ids_perturbed = np.concatenate([random_remote_prefix, local_token_ids])
                                if self.global_prefix_config != 'no_global_prefix':
                                    assert len(token_ids_perturbed) == i
                                prompt_token_ids.append(list(token_ids_perturbed))

                                custom_input = custom_tokenize_string(token_ids_perturbed, attention, len(token_ids_perturbed))
                                for attribute in custom_input:
                                    custom_input[attribute] = custom_input[attribute].to(args.device)
                                
                                # print(token_ids)
                                # print(i, prefix_length, len(token_ids))
                                # print("Random prefix full", random_remote_prefix_full)
                                # print("global", random_remote_prefix)
                                # print("local", local_token_ids)
                                # print(prompt_token_ids[-1][:i-prefix_length], prompt_token_ids[-1][i-prefix_length:])
                                # print(random_remote_prefix, local_token_ids)
                                # print(i, prefix_length)
                                # print(len(token_ids), len(random_remote_prefix), len(local_token_ids))
                                # print(len(token_ids_perturbed))
                                # print(prompt_token_ids[-1])
                                # print([len(elem) for elem in dataset_token_ids_sufficient])
                                # print(custom_input)
                                
                                # print(custom_input['input_ids'])
                                hf_output = model.generate(**custom_input, 
                                                            max_new_tokens=self.max_new_tokens,
                                                            do_sample=False,
                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                            top_k=None,
                                                            top_p=None,

                                )
                                # model_responses.append(hf_output)

                                predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                                ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + self.max_new_tokens]
                                min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                                negative_log_prob = []
                                for new_token_idx in range(min_length):
                                    all_token_probs = torch.nn.functional.softmax(hf_output['scores'][new_token_idx][0], dim=0).cpu().numpy()
                                    token_prob = all_token_probs[ground_truth_token_ids[new_token_idx]] # loss w.r.t. ground truth
                                    negative_log_prob.append(-np.log(token_prob + EPS))

                                
                                if(min_length == 0):
                                    continue
                                predicted_token_ids = predicted_token_ids[:min_length]
                                ground_truth_token_ids = ground_truth_token_ids[:min_length]
                                negative_log_prob = negative_log_prob[:min_length]
                                
                                # store values
                                if i == prefix_length:
                                    msp_prefix_length.append("full")
                                else:
                                    msp_prefix_length.append(prefix_length)
                                random_index.append(rand_idx)
                                prompt_ids.append(i)
                                length_token_ids_all.append(i)
                                original_prompt_token_ids.append(list(token_ids[:i]))
                                ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                                prompt_token_ids_all.append(prompt_token_ids[-1])
                                example_ids.append(index)
                                generated_token_ids_all.append(list(predicted_token_ids))
                                generated_token_negative_log_prob_all.append(negative_log_prob)

                                if prefix_length == i:
                                    break

                            
                    else:
                        # length_token_ids_all.append(max(0, i+1-num_pad_tokens))
                        prompt_token_ids.append(list(token_ids[:i]))
                        custom_input = custom_tokenize_string(token_ids, attention, i)
                        for attribute in custom_input:
                            custom_input[attribute] = custom_input[attribute].to(args.device)
                        
                        
                        hf_output = model.generate(**custom_input, 
                                                    max_new_tokens=self.max_new_tokens,
                                                    do_sample=False,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    top_k=None,
                                                    top_p=None,
                        )


                        # model_responses.append(hf_output)
                        predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                        ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + self.max_new_tokens]
                        min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                        if(min_length == 0):
                            continue
                        predicted_token_ids = predicted_token_ids[:min_length]
                        ground_truth_token_ids = ground_truth_token_ids[:min_length]
                        
                        # store values
                        length_token_ids_all.append(i)
                        ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                        prompt_token_ids_all.append(prompt_token_ids[-1])
                        example_ids.append(index)
                        generated_token_ids_all.append(list(predicted_token_ids))


                # for i, output in enumerate(model_responses):
                #     predicted_token_ids = output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[i]):]

                #     ground_truth_token_ids = token_ids[len(prompt_token_ids[i]): len(prompt_token_ids[i]) + self.max_new_tokens]
                #     min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                #     if(min_length == 0):
                #         continue
                #     predicted_token_ids = predicted_token_ids[:min_length]
                #     ground_truth_token_ids = ground_truth_token_ids[:min_length]
                #     ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                #     prompt_token_ids_all.append(prompt_token_ids[i])
                #     example_ids.append(index)
                #     generated_token_ids_all.append(list(predicted_token_ids))

                # if index == 5:
                #     break

                # print(self.global_prefix_config)
                # print(example_ids)
                # fa

            result = {
                "example_ids": example_ids,
                # "prompts": prompts,
                "prompt_token_ids": prompt_token_ids_all,
                # "generated_texts": generated_texts,
                "generated_token_ids": generated_token_ids_all,
                # "ground_truths": ground_truths,
                "ground_truth_token_ids": ground_truth_token_ids_all,
                "length_input_tokens": length_token_ids_all
            }


            if self.compute_msp:
                result['msp_prefix_length'] = msp_prefix_length
                result['original_prompt_token_ids'] = original_prompt_token_ids
                result['prompt_ids'] = prompt_ids
                result['random_index'] = random_index
                result['target_token_negative_log_prob_list'] = generated_token_negative_log_prob_all

                    
            result = pd.DataFrame(result)
            # print(result)
            # print(result.shape)
            result['eval_dataset'] = eval_dataset
            result['epoch'] = state.epoch

            # replace newline in prompts and generated_texts
            # result['prompts'] = result['prompts'].str.replace("\n", "[newline]")
            # result['generated_texts'] = result['generated_texts'].str.replace("\n", "[newline]")

            if(not os.path.exists(f"{args.output_dir}/text_generation_result.csv")):
                result.to_csv(f"{args.output_dir}/text_generation_result.csv", index=False)
            else:
                result.to_csv(f"{args.output_dir}/text_generation_result.csv", mode='a', header=False, index=False)



def compute_metrics(grammarCallback, selected_token_ids):
    clf_metrics = evaluate.combine(["accuracy"])

    def compute_metrics_for_grammar(eval_preds):

        processed_logits, labels = eval_preds

        preds = processed_logits[:, :, 0] # logits shape: (batch_size, seq_len)
        predicted_token_prob = processed_logits[:, :, 1] # predicted_token_prob shape: (batch_size, seq_len)
        target_token_prob = processed_logits[:, :, 2] # target_token_prob shape: (batch_size, seq_len)
        selected_token_probs = processed_logits[:, :, 3:] # selected_token_probs shape: (batch_size, seq_len, len(selected_token_ids))
        
        # print(f"Preds shape: {preds.shape}")
        # print(f"Predicted token prob shape: {predicted_token_prob.shape}")
        # print(f"Correct token prob shape: {target_token_prob.shape}")
        # print(f"Selected token probs shape: {selected_token_probs.shape}")
        # print(f"Labels shape: {labels.shape}")


        # Shift position of labels. pred position is already shifted in preprocess_logits_for_metrics
        shift_labels = labels[..., 1:]
        # print(f"Shift labels shape: {shift_labels.shape}")

        mask = shift_labels != -100

        # accuracy
        preds_flatten = preds.flatten()
        shift_labels_flatten = shift_labels.flatten()
        # result = clf_metrics.compute(predictions=preds_flatten, references=shift_labels_flatten)
        result = clf_metrics.compute(predictions=preds_flatten[mask.flatten()], references=shift_labels_flatten[mask.flatten()])
        
        # average predicted token prob per token per sequence
        result["predicted_token_prob"] = np.mean(predicted_token_prob[mask])

        # average correct token prob per token per sequence
        result["target_token_prob"] = np.mean(target_token_prob[mask])

        # average total probability mass per sequence per token
        total_prob_mass = np.sum(selected_token_probs, axis=-1)
        result["total_prob_mass"] = np.mean(total_prob_mass[mask])

        store_result_dict = {}
        store_result_dict["label_id"] = shift_labels_flatten
        store_result_dict["pred_id"] = preds_flatten
        store_result_dict["mask"] = mask.flatten()
        store_result_dict["predicted_token_prob"] = predicted_token_prob.flatten()
        store_result_dict["target_token_prob"] = target_token_prob.flatten()
        EPS = 1e-12
        store_result_dict['target_token_negative_log_prob'] = -np.log(target_token_prob.flatten() + EPS)
        store_result_dict["total_prob_mass"] = total_prob_mass.flatten()
        for i, token_id in enumerate(selected_token_ids):
            store_result_dict[f'token_prob_{token_id}'] = selected_token_probs[..., i].flatten()
        grammarCallback.store_result_dict = store_result_dict

        return result

    return compute_metrics_for_grammar

class GrammarCallback(TrainerCallback):


    def __init__(self, base_config, trainer, tokenizer, dataset, incontext_common_prefix_len):
        self.base_config = base_config
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.inference_only_mode = base_config['inference_only_mode']
        self.incontext_input = base_config['incontext_input']
        self.incontext_common_prefix_len = incontext_common_prefix_len
        self.decoding_cache = {}
        self.store_result_dict = None
        self.intermediate_result = pd.DataFrame()
        if self.base_config['memorization_intervention'] is not None:
            self.__config_check()
        

    def __config_check(self):
        with open(f"{self.base_config['memorization_intervention']}/args.pkl", "rb") as f:
            memorization_config = pickle.load(f)

            ignore_keys = ['comment', 'generate_text', 'compute_msp', 'global_prefix_config', 'use_deepspeed', 'memorization_intervention', 'memorization_approach']
            for key in memorization_config:
                if key in ignore_keys:
                    continue
                assert memorization_config[key] == self.base_config[key], key

            self.df_string_memorization = pd.read_csv(f"{self.base_config['memorization_intervention']}/string_memorization.csv")
            assert self.base_config['memorization_approach'] in self.df_string_memorization['approach'].unique(), f"Memorization approach {self.base_config['memorization_approach']} not found in string memorization csv"
            self.df_string_memorization = self.df_string_memorization[
                (self.df_string_memorization['approach'] == self.base_config['memorization_approach']) & 
                (self.df_string_memorization['epoch'] == self.df_string_memorization['epoch'].max()) &
                (self.df_string_memorization['metric'] == 'target_token_negative_log_prob')
            ]
            print(self.df_string_memorization[['sample_id', 'parameter', 'epoch']])
            
        
    def on_evaluate(self, args, state, control, **kwargs):
        # action performed after compute_metrics
        
        eval_dataset = None
        for key in state.log_history[-1].keys():
            if(key.startswith("eval") and key.endswith("loss")):
                eval_dataset = key[5:-5]
                break

        assert eval_dataset is not None,f"{eval_dataset} not found"

        
        result_dict = self.store_result_dict
        result = pd.DataFrame(result_dict)
        result['epoch'] = state.epoch if state.epoch is not None else 0
        result['global_step'] = state.global_step
        result['eval_dataset'] = eval_dataset
        result['pred_id'] = result['pred_id'].astype(int)
        result['label_id'] = result['label_id'].astype(int)
        result['index_token_ids'] = result.index
        
        
        assert eval_dataset in self.dataset
        loader = DataLoader(self.dataset[eval_dataset], batch_size=args.per_device_eval_batch_size, shuffle=False)
        length_token_ids = []
        for batch in loader:
            batch_token_ids = batch['input_ids'].cpu().numpy()
            for token_ids in batch_token_ids:
                num_pad_tokens = 0
                for token_id in token_ids:
                    if token_id == self.tokenizer.pad_token_id or token_id == self.tokenizer.bos_token_id:
                        num_pad_tokens += 1
                    else:
                        break
                for i in range(len(token_ids)-1):
                    length_token_ids.append(max(0, i+1-num_pad_tokens))
        result['length_input_tokens'] = length_token_ids
        
        if self.incontext_input:
            result = result[result['length_input_tokens'] >= self.incontext_common_prefix_len]
        result['correct'] = result['pred_id'] == result['label_id']
        

        # storing results of training sequences
        self.intermediate_result = pd.concat([self.intermediate_result, result[result['eval_dataset'] == 'train_sequences']]).copy()
    
        # store once, for local rank 0
        if(args.local_rank != 0):
            return
        

        # store the result
        if(not os.path.exists(f"{args.output_dir}/grammar_eval_result.csv")):
            result.to_csv(f"{args.output_dir}/grammar_eval_result.csv", index=False)
        else:
            result.to_csv(f"{args.output_dir}/grammar_eval_result.csv", mode='a', header=False, index=False)

        
        # average per epoch
        # result_average = result[~result['label_id'].isin([-100, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id])].groupby(['eval_dataset', 'epoch']).aggregate({
        result_average = result[~result['label_id'].isin([-100])].groupby(['eval_dataset', 'epoch']).aggregate({
            'target_token_negative_log_prob': 'mean',
            'correct': 'mean',
            'total_prob_mass': 'mean'
        }).reset_index()
        if(not os.path.exists(f"{args.output_dir}/grammar_eval_result_average.csv")):
            result_average.to_csv(f"{args.output_dir}/grammar_eval_result_average.csv", index=False)
        else:
            result_average.to_csv(f"{args.output_dir}/grammar_eval_result_average.csv", mode='a', header=False, index=False)


    def on_epoch_end(self, args, state, control, **kwargs):
        if self.intermediate_result.shape[0] == 0 or self.incontext_input:
            return
        
        if self.base_config['memorization_intervention'] is None:
            return


        assert self.intermediate_result['epoch'].nunique() == 1
        assert self.intermediate_result['eval_dataset'].nunique() == 1
        
        if args.local_rank == 0:
            print("Epoch:", state.epoch)
            print(self.trainer.train_dataset)

        self.intermediate_result = self.intermediate_result[~self.intermediate_result['label_id'].isin([-100, self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id])]
        self.intermediate_result['sample_id'] = (self.intermediate_result['length_input_tokens'] == 0).cumsum() - 1
        
        self.intermediate_result = self.intermediate_result.groupby(['sample_id']).aggregate({
            'target_token_negative_log_prob': 'mean',
        }).reset_index()
        
        
        # merge
        assert self.intermediate_result.shape[0] == self.df_string_memorization.shape[0]
        self.intermediate_result = self.intermediate_result.merge(self.df_string_memorization[['sample_id', 'parameter', 'distance_from_test']], how='left', on=['sample_id'])

        if args.local_rank == 0:
            print(self.intermediate_result)
        
        # memorized strings
        ignore_sample_ids = self.intermediate_result[self.intermediate_result['target_token_negative_log_prob'] <= self.intermediate_result['parameter']]['sample_id'].values
        assert len(ignore_sample_ids) == len(set(ignore_sample_ids))
        self.intermediate_result = pd.DataFrame()
        
        if len(ignore_sample_ids) != self.dataset['train_sequences'].num_rows:
            selected_data = [self.dataset['train_sequences'][i] for i in range(len(self.dataset['train_sequences'])) if i not in ignore_sample_ids]
            modified_dataset = Dataset.from_dict({k: [dic[k] for dic in selected_data] for k in selected_data[0].keys()})
            self.trainer.train_dataset = modified_dataset
            if args.local_rank == 0:
                print(f"Pruned {len(ignore_sample_ids)} memorization samples.")
        else:
            if args.local_rank == 0:
                print(f"All memorization samples pruned.")
            control.should_training_stop = True
        return


def preprocess_logits(tokenizer, selected_token_ids):
    softmax = torch.nn.Softmax(dim=-1)
    def preprocess_logits_for_metrics(logits, labels):
        """
                Return a tensor (max prob token ids, predicted token prob, correct token prob, selected token probs)
        """
        # shift position of logits
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # logits shape: (batch_size, seq_len, vocab_size)
        pred_ids = torch.argmax(logits, dim=-1) # pred_ids shape: (batch_size, seq_len)
        softmax_prob = softmax(logits) # softmax_prob shape: (batch_size, seq_len, vocab_size)
        
        # softmax prob of predicted tokens
        pred_token_probs = softmax_prob.gather(-1, pred_ids.unsqueeze(-1)) # pred_token_probs shape: (batch_size, seq_len, 1)
        # print(f"Pred token probs shape: {pred_token_probs.shape}")

        
        # softmax prob of correct tokens
        labels = torch.where(labels == -100, tokenizer.pad_token_id, labels) # repace -100 with tokenizer.pad_token_id
        target_token_probs = softmax_prob.gather(-1, labels.unsqueeze(-1)) # target_token_probs shape: (batch_size, seq_len, 1)
        # print(f"Correct token probs shape: {target_token_probs.shape}")

        # softmax prob of selected tokens
        selected_token_probs = softmax_prob[:, :, torch.tensor(selected_token_ids).to(logits.device)] # selected_token_probs shape: (batch_size, seq_len, len(selected_token_ids))
        
        # concatenate: pred_ids followed by selected_token_probs
        pred_ids = pred_ids.unsqueeze(-1) # pred_ids shape: (batch_size, seq_len, 1)
        return_ids = torch.cat([pred_ids,
                                pred_token_probs,
                                target_token_probs,
                                selected_token_probs], dim=-1) # pred_ids shape: (batch_size, seq_len, 1+len(selected_token_ids))

        # print(f"Return id shape: {return_ids.shape}")
        return return_ids
    
    return preprocess_logits_for_metrics





def compute_inference_results(model, 
                          tokenizer, 
                          dataset, 
                          selected_token_ids,
                          incontext_common_prefix_len,
                          store_path,
                          device,
                          batch_size=4,
):

    def remove_eos_and_add_bos(token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        length = len(token_ids_raw)
        # remove eos and other unwanted tokens
        for i, (t, a) in enumerate(zip(token_ids_raw, attentions_raw)):
            if(t in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]):
                # print(i)
                continue
            token_ids.append(t)
            attentions.append(a)

        # add bos token
        token_ids.insert(0, tokenizer.bos_token_id)
        attentions.insert(0, 0)
        return token_ids, attentions

    def find_cut_off(input_ids_prompt):
        # find common_incontext_prefix part
        non_interested_token_count = 0
        for token_id in input_ids_prompt:
            if token_id not in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
                break
            non_interested_token_count += 1
        # return non_interested_token_count + incontext_common_prefix_len
        return non_interested_token_count

    def cached_generation_common_prefix(input_ids_prompt, input_attn_mask_prompt):
        custom_input = custom_tokenize_string(input_ids_prompt, input_attn_mask_prompt, len(input_ids_prompt))
        for attribute in custom_input:
            custom_input[attribute] = custom_input[attribute].to(device)
        output = model.generate(**custom_input,
                                max_new_tokens=1,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                top_k=None,
                                top_p=None,
                                use_cache=True,
                                output_scores=True,
                                return_dict_in_generate=True)
        return output
    
    def expand_generation_output(single_output, batch_size):
        """
        Expands a Hugging Face generation output from batch size 1 to a new batch size N.

        Args:
            single_output: output from `model.generate(...)` with batch_size=1
            batch_size: target batch size (number of duplicates)

        Returns:
            A modified output object with batch_size = N
        """
        expanded_output = deepcopy(single_output)

        # Expand sequences [1, seq_len] → [batch_size, seq_len]
        if hasattr(single_output, "sequences"):
            expanded_output.sequences = single_output.sequences.expand(batch_size, -1).clone()

        # Expand scores: list of [1, vocab_size] → list of [batch_size, vocab_size]
        if hasattr(single_output, "scores") and single_output.scores is not None:
            expanded_output.scores = [
                score.expand(batch_size, -1).clone() for score in single_output.scores
            ]

        # Expand past_key_values: tuple of (key, value) → each [1, heads, seq_len, head_dim]
        if hasattr(single_output, "past_key_values") and single_output.past_key_values is not None:
            expanded_output.past_key_values = tuple(
                (
                    key.expand(batch_size, -1, -1, -1).clone(),
                    value.expand(batch_size, -1, -1, -1).clone()
                )
                for key, value in single_output.past_key_values
            )

        return expanded_output

    def prepare_model_input(tokens, attention):
        custom_input = transformers.tokenization_utils_base.BatchEncoding()
        custom_input['input_ids'] = torch.tensor(tokens)
        custom_input['attention_mask'] = torch.tensor(attention)

        # print(custom_input['input_ids'].shape)
        return custom_input

    def pad_batch(batch, max_length):
        batch_processed = {
            "input_ids": [],
            "attention_mask": []
        }
        for input_ids, attention_mask in batch:
            # print(len(input_ids), len(attention_mask))
            num_pad = max_length - len(input_ids)
            batch_processed["input_ids"].append(input_ids + [tokenizer.eos_token_id] * num_pad)
            batch_processed["attention_mask"].append(attention_mask + [0] * num_pad)
            # print(len(batch_processed["input_ids"][-1]), len(batch_processed["attention_mask"][-1]))

        return batch_processed



    EPS = 1e-12
    def guided_generation(batch_target, max_length):

        effective_batch_size = len(batch_target['input_ids'])
        softmax = torch.nn.Softmax(dim=-1)

        selected_token_probs = [[] for _ in range(effective_batch_size)]
        pred_token_ids = [[] for _ in range(effective_batch_size)]
        target_token_probs = [[] for _ in range(effective_batch_size)]
        target_token_negative_log_probs = [[] for _ in range(effective_batch_size)]
        predicted_token_probs = [[] for _ in range(effective_batch_size)]

        input_ids_unexpanded = deepcopy(initialization['input_ids_incontext'])
        attention_mask_unexpanded = deepcopy(initialization['attention_mask_incontext'])

        output_unexpanded = deepcopy(initialization['cached_output'])
        output = expand_generation_output(output_unexpanded, effective_batch_size)
        
        input_ids = [deepcopy(input_ids_unexpanded) for _ in range(effective_batch_size)]
        attention_mask = [deepcopy(attention_mask_unexpanded) for _ in range(effective_batch_size)]
        
        past_key_values = output['past_key_values']
        
        for idx in range(max_length):

            softmax_probs_all = softmax(output['scores'][0]).cpu().numpy()
            pred_token_ids_all = torch.argmax(output['scores'][0], dim=-1).cpu().numpy()
            
            for idx_batch in range(effective_batch_size):
                token_prob = softmax_probs_all[idx_batch][batch_target['input_ids'][idx_batch][idx]]
                target_token_negative_log_probs[idx_batch].append(-np.log(token_prob + EPS))
                target_token_probs[idx_batch].append(token_prob)
                
                pred_token_ids[idx_batch].append(pred_token_ids_all[idx_batch])
                predicted_token_probs[idx_batch].append(softmax_probs_all[idx_batch].max())

                selected_token_prob = {}
                for selected_token_id in selected_token_ids:
                    selected_token_prob[selected_token_id] = softmax_probs_all[idx_batch][selected_token_id].item()
                selected_token_probs[idx_batch].append(selected_token_prob)
                

            
            # append to incontext_prompt (in a batch fashion)
            for idx_batch in range(effective_batch_size):
                input_ids[idx_batch].append(batch_target['input_ids'][idx_batch][idx])
                attention_mask[idx_batch].append(batch_target['attention_mask'][idx_batch][idx])
                
            
            custom_input = prepare_model_input(input_ids, attention_mask)
            for attribute in custom_input:
                custom_input[attribute] = custom_input[attribute].to(device)
            
            
            output = model.generate(**custom_input,
                                    max_new_tokens=1,
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    top_k=None,
                                    top_p=None,
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    return_dict_in_generate=True,
                                    output_scores=True
            )

            past_key_values = output['past_key_values']

            
        return target_token_negative_log_probs, target_token_probs, selected_token_probs, pred_token_ids, predicted_token_probs


    
    

    representative_eval_dataset = "incontext_common_prefix"
    if representative_eval_dataset in dataset:
        assert dataset[representative_eval_dataset].num_rows == 1
        input_ids_incontext = dataset[representative_eval_dataset][0]['input_ids'].tolist()
        attention_mask_incontext = dataset[representative_eval_dataset][0]['attention_mask'].tolist()
        input_ids_incontext, attention_mask_incontext = remove_eos_and_add_bos(input_ids_incontext, attention_mask_incontext)
    else:
        input_ids_incontext = [tokenizer.bos_token_id]
        attention_mask_incontext = [0]    
 
    # print(input_ids_incontext)
    # print(attention_mask_incontext)
    
    initialization = {}
    initialization['cached_output'] = cached_generation_common_prefix(input_ids_incontext, attention_mask_incontext)
    initialization['input_ids_incontext'] = input_ids_incontext
    initialization['attention_mask_incontext'] = attention_mask_incontext


    for eval_dataset in dataset:
        list_grammar_eval_result = []
        if eval_dataset == representative_eval_dataset:
            continue
        # print(eval_dataset, dataset[eval_dataset].num_rows)
        for i in tqdm(range(0, dataset[eval_dataset].num_rows, batch_size)):
            batch = dataset[eval_dataset][i:i + batch_size]
            
            batch_processed = []
            max_length = 0
            length_orig = []
            for j in range(len(batch['input_ids'])):
                input_ids = batch['input_ids'][j].tolist()
                attention_mask = batch['attention_mask'][j].tolist()
                cut_off = find_cut_off(input_ids)
                input_ids_target = input_ids[cut_off:]
                attention_mask_target = attention_mask[cut_off:]
                max_length = max(max_length, len(input_ids_target))
                length_orig.append(len(input_ids_target))
                batch_processed.append((input_ids_target, attention_mask_target))
            
            batch_processed_with_pad = pad_batch(batch_processed, max_length)

             
                         
            target_token_negative_log_probs, target_token_probs, output_selected_token_probs, pred_token_ids, predicted_token_probs = guided_generation(batch_processed_with_pad, max_length)
            
            
            for idx_batch in range(len(length_orig)):
                j = 0
                for label_id, \
                    pred_id, \
                    predicted_token_prob, \
                    target_token_prob, \
                    target_token_negative_log_prob, \
                    selected_token_probs in zip(
                        batch_processed_with_pad['input_ids'][idx_batch],
                        pred_token_ids[idx_batch],
                        predicted_token_probs[idx_batch],
                        target_token_probs[idx_batch],
                        target_token_negative_log_probs[idx_batch],
                        output_selected_token_probs[idx_batch]
                    ):

                    if j >= length_orig[idx_batch]:
                        break

                    result = {}
                    result['label_id'] = label_id
                    result['pred_id'] = pred_id
                    result['predicted_token_prob'] = predicted_token_prob
                    result['target_token_prob'] = target_token_prob
                    result['target_token_negative_log_prob'] = target_token_negative_log_prob
                    sum_selected_token_probs = 0
                    for selected_token_id in selected_token_ids:
                        result[f"token_prob_{selected_token_id}"] = selected_token_probs[selected_token_id]
                        sum_selected_token_probs += selected_token_probs[selected_token_id]
                    result['total_prob_mass'] = sum_selected_token_probs
                    result['mask'] = None
                    result['epoch'] = 0
                    result['global_step'] = 0
                    result['eval_dataset'] = eval_dataset
                    result['index_token_ids'] = i
                    result['length_input_tokens'] = incontext_common_prefix_len + j
                    result['correct'] = result['label_id'] == result['pred_id']

                    list_grammar_eval_result.append(result)
                    j += 1
        
        df_grammar_eval_result = pd.DataFrame(list_grammar_eval_result)
        os.system(f"mkdir -p {store_path}")
        if(not os.path.exists(f"{store_path}/grammar_eval_result.csv")):
            df_grammar_eval_result.to_csv(f"{store_path}/grammar_eval_result.csv", index=False)
        else:
            df_grammar_eval_result.to_csv(f"{store_path}/grammar_eval_result.csv", mode='a', header=False, index=False)
        
                
    return





def text_generation(
    model,
    tokenizer,
    dataset,
    comment,
    device,
    output_dir,
    eval_dataset = "train_sequences",
    max_new_tokens = 1,
    compute_msp = True,
    local_prefix_length_list = [5, 10, 20],
    skip_tokens=0,
    generation_interval=1,
    selective_samples=True
):


    def remove_eos(token_ids_raw, attentions_raw):
        token_ids = []
        attentions = []
        length = len(token_ids_raw)
        for i, (t, a) in enumerate(zip(token_ids_raw, attentions_raw)):
            if(t == tokenizer.eos_token_id and i < length - 1):
                continue
            token_ids.append(t)
            attentions.append(a)
        return token_ids, attentions

    EPS = 1e-12
    ground_truth_token_ids_all = []
    prompt_token_ids_all = []
    example_ids = []
    generated_token_ids_all = []
    length_token_ids_all = []


    random_index_list = None

    if compute_msp:
        msp_prefix_length = []
        original_prompt_token_ids = []
        prompt_ids = []
        random_index = []
        generated_token_negative_log_prob_all = []
        np.random.seed(0)
        if not selective_samples:
            max_samples_considered = 3,
            if dataset[eval_dataset].shape[0] > max_samples_considered:
                random_index_list = np.random.choice(
                    dataset[eval_dataset].shape[0], size=max_samples_considered, replace=False
                )
            else:
                random_index_list = np.arange(dataset[eval_dataset].shape[0])
        else:
            sequence_to_index_map = {}
            for index, token_id in enumerate(dataset[eval_dataset]['input_ids']):
                token_id = tuple(token_id.cpu().numpy())
                if token_id not in sequence_to_index_map:
                    sequence_to_index_map[token_id] = []
                sequence_to_index_map[token_id].append(index)

        
            sequence_freq = {}
            for sequence in sequence_to_index_map:
                sequence_freq[sequence] = len(sequence_to_index_map[sequence])
            sequence_freq = dict(sorted(sequence_freq.items(), key=lambda x: x[1], reverse=True))

            
            # take max, median, and min
            max_idx = 0
            min_idx = -1
            median_idx = len(sequence_freq) // 2
            random_index_list = [sequence_to_index_map[list(sequence_freq.keys())[max_idx]][0],
                                 sequence_to_index_map[list(sequence_freq.keys())[median_idx]][0],
                                 sequence_to_index_map[list(sequence_freq.keys())[min_idx]][0]]
            
        print(f"Random index list: {random_index_list}")
            

    dataset_token_ids = []
    for index in tqdm(range(len(dataset[eval_dataset]))):
        token_ids_raw, attention_raw = dataset[eval_dataset]['input_ids'].tolist()[index], dataset[eval_dataset]['attention_mask'].tolist()[index]
        token_ids, attention = remove_eos(token_ids_raw, attention_raw) # this turns out to be a good idea
        token_ids = np.array(token_ids)
        dataset_token_ids.append(token_ids)


    for index in tqdm(range(len(dataset[eval_dataset]))):
        if compute_msp and index not in random_index_list:
            continue

        token_ids_raw, attention_raw = dataset[eval_dataset]['input_ids'].tolist()[index], dataset[eval_dataset]['attention_mask'].tolist()[index]
        token_ids, attention = remove_eos(token_ids_raw, attention_raw) # this turns out to be a good idea
        token_ids = np.array(token_ids)
        
        

        prompt_token_ids = []
        # model_responses = []
        token_length = token_ids.shape[0]    
        for i in range(1, token_ids.shape[0] - 1):
            
            if i % generation_interval != 0 or i + max_new_tokens > token_length or i <= skip_tokens:
                continue
            

            if compute_msp:
                assert max_new_tokens == 1
                for prefix_length in local_prefix_length_list + [i]:
                    if prefix_length > i:
                        continue
                    # print("Prefix length:", prefix_length)
                
                    for rand_idx in range(5):
                        dataset_token_ids_sufficient = []
                        for token_ids_temp in dataset_token_ids:
                            if len(token_ids_temp) >= i - prefix_length:
                                dataset_token_ids_sufficient.append(token_ids_temp)

                        if len(dataset_token_ids_sufficient) == 0:
                            continue
                    
                        random_remote_prefix_full = dataset_token_ids_sufficient[np.random.choice(len(dataset_token_ids_sufficient))]
                        random_remote_prefix = random_remote_prefix_full[:i-prefix_length].copy()
                        local_token_ids = token_ids[i-prefix_length:i]
                        token_ids_perturbed = np.concatenate([random_remote_prefix, local_token_ids])
                        prompt_token_ids.append(list(token_ids_perturbed))

                        custom_input = custom_tokenize_string(token_ids_perturbed, attention, i)
                        for attribute in custom_input:
                            custom_input[attribute] = custom_input[attribute].to(device)
                        
                        
                        hf_output = model.generate(**custom_input, 
                                                    max_new_tokens=max_new_tokens,
                                                    do_sample=False,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    top_k=None,
                                                    top_p=None,

                        )
                        

                        predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                        ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + max_new_tokens]
                        min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                        negative_log_prob = []
                        for new_token_idx in range(min_length):
                            all_token_probs = torch.nn.functional.softmax(hf_output['scores'][new_token_idx][0], dim=0).cpu().numpy()
                            token_prob = all_token_probs[ground_truth_token_ids[new_token_idx]] # loss w.r.t. ground truth
                            negative_log_prob.append(-np.log(token_prob + EPS))

                        
                        if(min_length == 0):
                            continue
                        predicted_token_ids = predicted_token_ids[:min_length]
                        ground_truth_token_ids = ground_truth_token_ids[:min_length]
                        negative_log_prob = negative_log_prob[:min_length]
                        
                        # store values
                        if i == prefix_length:
                            msp_prefix_length.append("full")
                        else:
                            msp_prefix_length.append(prefix_length)
                        random_index.append(rand_idx)
                        prompt_ids.append(i)
                        length_token_ids_all.append(i)
                        original_prompt_token_ids.append(list(token_ids[:i]))
                        ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                        prompt_token_ids_all.append(prompt_token_ids[-1])
                        example_ids.append(index)
                        generated_token_ids_all.append(list(predicted_token_ids))
                        generated_token_negative_log_prob_all.append(negative_log_prob)

                        if prefix_length == i:
                            break
            else:
                # length_token_ids_all.append(max(0, i+1-num_pad_tokens))
                prompt_token_ids.append(list(token_ids[:i]))
                custom_input = custom_tokenize_string(token_ids, attention, i)
                for attribute in custom_input:
                    custom_input[attribute] = custom_input[attribute].to(device)
                
                
                hf_output = model.generate(**custom_input, 
                                            max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id,
                                            top_k=None,
                                            top_p=None,
                )


                # model_responses.append(hf_output)
                predicted_token_ids = hf_output['sequences'][-1].cpu().numpy()[len(prompt_token_ids[-1]):]
                ground_truth_token_ids = token_ids[len(prompt_token_ids[-1]): len(prompt_token_ids[-1]) + max_new_tokens]
                min_length = min(len(predicted_token_ids), len(ground_truth_token_ids))
                if(min_length == 0):
                    continue
                predicted_token_ids = predicted_token_ids[:min_length]
                ground_truth_token_ids = ground_truth_token_ids[:min_length]
                
                # store values
                length_token_ids_all.append(i)
                ground_truth_token_ids_all.append(list(ground_truth_token_ids))
                prompt_token_ids_all.append(prompt_token_ids[-1])
                example_ids.append(index)
                generated_token_ids_all.append(list(predicted_token_ids))


        

    result = {
        "example_ids": example_ids,
        "prompt_token_ids": prompt_token_ids_all,
        "generated_token_ids": generated_token_ids_all,
        "ground_truth_token_ids": ground_truth_token_ids_all,
        "length_input_tokens": length_token_ids_all
    }


    if compute_msp:
        result['msp_prefix_length'] = msp_prefix_length
        result['original_prompt_token_ids'] = original_prompt_token_ids
        result['prompt_ids'] = prompt_ids
        result['random_index'] = random_index
        result['target_token_negative_log_prob_list'] = generated_token_negative_log_prob_all
        
            
    result = pd.DataFrame(result)
    result['eval_dataset'] = eval_dataset
    result['comment'] = comment


    if(not os.path.exists(f"{output_dir}/text_generation_result.csv")):
        result.to_csv(f"{output_dir}/text_generation_result.csv", index=False)
    else:
        result.to_csv(f"{output_dir}/text_generation_result.csv", mode='a', header=False, index=False)

    # return result




class PruneDatasetCallback(TrainerCallback):
    def __init__(self, trainer, dataset, base_config):
        self.trainer = trainer
        self.dataset = dataset
        self.base_config = base_config
        self.__config_check()
        

    def __config_check(self):
        with open(f"{self.base_config['memorization_intervention']}/args.pkl", "rb") as f:
            memorization_config = pickle.load(f)

            ignore_keys = ['comment', 'generate_text', 'compute_msp', 'global_prefix_config', 'use_deepspeed']
            for key in memorization_config:
                if key in ignore_keys:
                    continue
                assert memorization_config[key] == self.base_config[key], key

            self.df_string_memorization = pd.read_csv(f"{self.base_config['memorization_intervention']}/string_memorization.csv")
        pass


    def on_epoch_end(self, args, state, control, **kwargs):
        # print("Epoch", state.epoch)
        # print(self.trainer.train_dataset)

        ignore_sample_ids = self.df_string_memorization[
            (self.df_string_memorization['epoch'] == state.epoch) &
            (self.df_string_memorization['approach'] == 'contextual_memorization') &
            (self.df_string_memorization['memorization_binary'] == True) &
            (self.df_string_memorization['eval_dataset'] == 'train_sequences') &
            (self.df_string_memorization['metric'] == 'target_token_negative_log_prob')
        ]['sample_id'].values
        assert len(ignore_sample_ids) == len(set(ignore_sample_ids))
        
        if len(ignore_sample_ids) == self.dataset.num_rows:
            control.should_training_stop = True
            return
        
        new_data = [self.dataset[i] for i in range(len(self.dataset)) if i not in ignore_sample_ids]
        modified_dataset = Dataset.from_dict({k: [dic[k] for dic in new_data] for k in new_data[0].keys()})
        self.trainer.train_dataset = modified_dataset

        print(f"Pruned {len(ignore_sample_ids)} memorization samples.")

        return
