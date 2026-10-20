import os
# specify cache
os.environ["HF_HUB_CACHE"] = "/path/to/shared/huggingface_cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/path/to/shared/huggingface_cache/datasets"



import sys
sys.path.append("../training")
from utils import get_tokenizer, get_data
from utils_plot_memorization import get_start_of_memorization, get_edit_distance, compare_with_nearest_test_sequence
import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
from exp_meta_data import model_revised_location
tqdm.pandas()
import argparse
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from ast import literal_eval


parser = argparse.ArgumentParser()
parser.add_argument("--conflict", action="store_true")
parser.add_argument("--data_source", type=str, default="edit_distance_results")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--memorization", action="store_true")
args_current = parser.parse_args()

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)




if not args_current.conflict:

    data_dir = "../training/results/"
    experiment_dict = {

        
        # "discriminative_individual/generalization/fine-tuning/without_checkpoints": None,
        
        # "inductive_bias/generalization/fine-tuning/without_checkpoints": None,
        # "inductive_bias/generalization/incontext_learning/base_model_performance": None,
        

        # "mat_large/generalization/fine-tuning/without_checkpoints": None,
        
        # # "edit_distance_results/generalization/fine-tuning/without_checkpoints": None,
        # # "edit_distance_results/generalization/incontext_learning/base_model_performance": None,
        

        # # "instruction_prompt/generalization/incontext_learning/base_model_performance": None,
        # # "memorization/random_strings/fine-tuning/without_checkpoints": None,
        # # "memorization/sensitivity/fine-tuning/without_checkpoints": None,
        # # "block_structure/generalization/fine-tuning/with_checkpoints": None,
        # # "sensitive_non_sensitive_tokens_fl/generalization/fine-tuning/without_checkpoints": None,
        
        # # "multilingual_with_mat/generalization/fine-tuning/without_checkpoints": None,
        

        # "dynamic_training_size/generalization/fine-tuning/without_checkpoints": None,
        # "dynamic_training_size/generalization/incontext_learning/base_model_performance": None,
        # # "dynamic_training_size_ood/generalization/fine-tuning/without_checkpoints": None,


        # "multilingual/generalization/fine-tuning/without_checkpoints": None,
        
        # "cfg_csg/generalization/fine-tuning/without_checkpoints": None,

        
        
        
        # "benchmark_revised/generalization/incontext_learning/base_model_performance": None,
        # "language_proficiency_all_eval/generalization/fine-tuning/without_checkpoints": None,
        # "ood_generalization/generalization/fine-tuning/without_checkpoints": None,
        # "token_generalization/generalization/fine-tuning/without_checkpoints": None,
        # "text_generation_with_mem/memorization/fine-tuning/without_checkpoints": None,
        # "memorization_more_test_data/generalization/incontext_learning/base_model_performance": None,
        # "memorization_more_test_data/generalization/fine-tuning/without_checkpoints": None,
        # "counterfactual_memorization_with_mitigation/generalization/fine-tuning/without_checkpoints": None,
        "counterfactual_memorization/memorization/fine-tuning/without_checkpoints": None,
        # "counterfactual_memorization/memorization/fine-tuning/test": None,
        "counterfactual_memorization/memorization/incontext_learning/base_model_performance": None,
        "counterfactual_memorization_nlp/memorization/fine-tuning/without_checkpoints": None,
        # "undertrained_tokens/generalization/incontext_learning/base_model_performance": None,
        # "undertrained_tokens/generalization/fine-tuning/without_checkpoints": None,
        # "entropy/generalization/incontext_learning/base_model_performance": None,
        # "entropy/generalization/fine-tuning/without_checkpoints": None,
        # "missing_icl_time/generalization/incontext_learning/base_model_performance": None,
        # "preg/generalization/fine-tuning/without_checkpoints": None,
        # "preg/generalization/incontext_learning/base_model_performance": None,
        # "pcsg/generalization/fine-tuning/without_checkpoints": None,
        # "pcsg/generalization/incontext_learning/base_model_performance": None,
        # "benchmark/generalization/incontext_learning/base_model_performance": None,
        # "benchmark/generalization/fine-tuning/without_checkpoints": None,
        # "text_generation/memorization/fine-tuning/without_checkpoints/msp": None,


        # "nlp_dataset/generalization/incontext_learning/base_model_performance": None,
        # "nlp_dataset/generalization/fine-tuning/without_checkpoints": None,
        
        

        # "revised/generalization/fine-tuning/with_checkpoints": None,
        # "revised/generalization/fine-tuning/with_checkpoints_more_models": None,
        # "revised/generalization/fine-tuning/without_checkpoints_rest_models": None,
        # "revised/generalization/fine-tuning/with_checkpoints_g2_grammar": None,
        # "acl_submission/generalization/fine-tuning/without_checkpoints_rest_models": None,
        # "acl_submission/generalization/fine-tuning/without_checkpoints": None,
        # "partial_ft/generalization/fine-tuning/with_checkpoints": None,
        # "partial_ft/generalization/fine-tuning/without_checkpoints": None,
        


        # "memorization/entropy_experiment/balanced_vs_skewed_cfg3b_eq_len": None,
        # "memorization/entropy_experiment/balanced_vs_skewed_cfg3b_eq_len_pythia_1b": None,
        # "memorization/entropy_experiment/balanced_vs_skewed_multiple_seeds": None,

        "mat_nlp/generalization/fine-tuning/without_checkpoints": None,
        # "mat_icml_2026_rebuttal/generalization/fine-tuning/without_checkpoints": None,
    }

else:
        
    data_dir_parent = f"../training/results/{args_current.data_source}/generalization"
    data_dir = f"{data_dir_parent}/incontext_learning/"
    experiment_dict = {
        "base_model_performance": ("Base Model", None, None, 0),
    }

    for considered_training_examples in [0, 16, 64, 256, 1024]:

        experiment_dict[f"ft_G1_icl_G1_test_G1_{considered_training_examples}"] = (f"Collab: FT G1 with n=", "G1", "G1", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G2_test_G1_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G2", "G1", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G2_test_G2_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G2", "G2", considered_training_examples)
        experiment_dict[f"ft_G2_icl_G2_test_G2_{considered_training_examples}"] = (f"Collab: FT G2 with n=", "G2", "G2", considered_training_examples)
        experiment_dict[f"ft_G2_icl_G1_test_G2_{considered_training_examples}"] = (f"Conflict: FT G2 with n=", "G1", "G2", considered_training_examples)
        experiment_dict[f"ft_G2_icl_G1_test_G1_{considered_training_examples}"] = (f"Conflict: FT G2 with n=", "G1", "G1", considered_training_examples)
        
        
        experiment_dict[f"ft_G1_icl_G55_test_G55_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G55", "G55", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G60_test_G60_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G60", "G60", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G70_test_G70_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G70", "G70", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G80_test_G80_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G80", "G80", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G90_test_G90_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G90", "G90", considered_training_examples)


        experiment_dict[f"ft_G1_icl_G1955_test_G1955_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1955", "G1955", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1960_test_G1960_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1960", "G1960", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1970_test_G1970_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1970", "G1970", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1980_test_G1980_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1980", "G1980", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1990_test_G1990_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1990", "G1990", considered_training_examples)


        experiment_dict[f"ft_G1_icl_G1_test_G1955_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G1955", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G1960_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G1960", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G1970_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G1970", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G1980_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G1980", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G1990_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G1990", considered_training_examples)


        experiment_dict[f"ft_G1_icl_G101_test_G101_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G101", "G101", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G102_test_G102_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G102", "G102", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G103_test_G103_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G103", "G103", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G104_test_G104_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G104", "G104", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G105_test_G105_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G105", "G105", considered_training_examples)


        experiment_dict[f"ft_G1_icl_G1_test_G101_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G101", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G102_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G102", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G103_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G103", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G104_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G104", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G1_test_G105_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G1", "G105", considered_training_examples)


        experiment_dict[f"ft_G4_icl_G4_test_G4_{considered_training_examples}"] = (f"Collab: FT G4 with n=", "G4", "G4", considered_training_examples)
        experiment_dict[f"ft_G4_icl_G2_test_G4_{considered_training_examples}"] = (f"Conflict: FT G4 with n=", "G2", "G4", considered_training_examples)
        experiment_dict[f"ft_G4_icl_G2_test_G2_{considered_training_examples}"] = (f"Conflict: FT G4 with n=", "G2", "G2", considered_training_examples)
        

        # experiment_dict[f"ft_G1_icl_G3_test_G1_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G3", "G1", considered_training_examples)
        experiment_dict[f"ft_G1_icl_G3_test_G3_{considered_training_examples}"] = (f"Conflict: FT G1 with n=", "G3", "G3", considered_training_examples)
    
    
        experiment_dict[f"ft_G1_G2_icl_G1_test_G1_{considered_training_examples}"] = (f"Conflict: FT G1_G2 with n=", "G1", "G1", considered_training_examples)
        experiment_dict[f"ft_G1_G2_icl_G2_test_G2_{considered_training_examples}"] = (f"Conflict: FT G1_G2 with n=", "G2", "G2", considered_training_examples)
        experiment_dict[f"ft_G1_G2_icl_G2_test_G1_{considered_training_examples}"] = (f"Conflict: FT G1_G2 with n=", "G2", "G1", considered_training_examples)
        experiment_dict[f"ft_G1_G2_icl_G1_test_G2_{considered_training_examples}"] = (f"Conflict: FT G1_G2 with n=", "G1", "G2", considered_training_examples)


if args_current.dir is not None:
    assert "/" in args_current.dir
    data_dir = ("/").join(args_current.dir.split("/")[:-1]) + "/"
    experiment = args_current.dir.split("/")[-1]
    experiment_dict = {
        experiment: None
    }



# filter = "Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0"
filter = ""


tokenizer_dict = {}



deleted_dir = []
for experiment in experiment_dict:
    if not os.path.isdir(f"{data_dir}/{experiment}"):
        continue
    if len(os.listdir(f"{data_dir}/{experiment}/")) == 0:
        continue

    for input_dir in tqdm(os.listdir(f"{data_dir}/{experiment}/"), disable=False):

        if not os.path.isdir(f"{data_dir}/{experiment}/{input_dir}"):
            continue
        if filter not in input_dir:
            continue

            
        # process wandb files
        store_filename_wandb_history = f"{data_dir}/{experiment}/{input_dir}/wandb_history.csv"
        store_filename_wandb_stats = f"{data_dir}/{experiment}/{input_dir}/wandb_stats.csv"

        if args_current.wandb and (not os.path.exists(store_filename_wandb_history) or not os.path.exists(store_filename_wandb_stats)):
            import wunderbar
            wandb_file_dir = f"{data_dir}/{experiment}/{input_dir}/wandb/latest-run/"
            if not os.path.isdir(wandb_file_dir):
                continue
            for wandb_file in os.listdir(wandb_file_dir):
                if wandb_file.startswith("run") and wandb_file.endswith(".wandb"):
                    print(f"Processing wandb file: {wandb_file_dir}/{wandb_file}")
                    records = wunderbar.parse_filepath(path=f"{wandb_file_dir}/{wandb_file}")
                    list_stats = []
                    list_history = []
                    for record in records:
                        
                        if record.type == "history": 
                            assert "item" in record.data
                            history = {}
                            for key in record.data["item"]:
                                history[key] = record.data["item"][key]
                            list_history.append(history)

                        if record.type == "stats":
                            run = record.data
                            assert "item" in run
                            assert "timestamp" in run
                            stats = {}
                            stats["timestamp"] = run["timestamp"]
                            for key in run["item"]:
                                stats[key] = run["item"][key]

                            list_stats.append(stats)  

                    df_history = pd.DataFrame(list_history)
                    df_stats = pd.DataFrame(list_stats)

                    df_history.to_csv(store_filename_wandb_history, index=False)
                    df_stats.to_csv(store_filename_wandb_stats, index=False)
                    break
                    
        if args_current.wandb:
            continue

        with open(f"{data_dir}/{experiment}/{input_dir}/args.pkl", 'rb') as f:
            args = pickle.load(f)

        
        # check if there is data
        result_file = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result.csv"
        if(not os.path.exists(result_file)):
            deleted_dir.append(input_dir)
            continue


        

        store_filename_optimal_checkpoint = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_optimal_checkpoint.csv"
        store_filename_optimal_auc = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_optimal_auc.csv"
        store_filename_optimal_auc_per_rule = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_optimal_auc_per_rule.csv"
        store_filename_optimal_test_loss = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_optimal_average.csv"
        store_filename_average = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_average.csv"
        store_filename_auc_average = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_average_auc.csv"
        store_filename_auc_average_per_rule = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_average_auc_per_rule.csv"
        store_filename_near_freq_test_comparison = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_near_freq_test_comparison.csv"
        store_filename_string_average = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_string_average.csv"
        store_filename_string_average_optimal = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_string_average_optimal_checkpoint.csv"
        store_filename_string_average_optimal_auc = f"{data_dir}/{experiment}/{input_dir}/grammar_eval_result_string_average_optimal_auc.csv"
        store_filename_memorization_results = f"{data_dir}/{experiment}/{input_dir}/string_memorization.csv"
        store_filename_discriminative_individual = f"{data_dir}/{experiment}/{input_dir}/discriminative_individual.csv"
        store_filename_discriminative_individual_optimal = f"{data_dir}/{experiment}/{input_dir}/discriminative_individual_optimal.csv"
        store_filename_discriminative_individual_root = f"{data_dir}/{experiment}/{input_dir}/discriminative_individual_root.csv"
        store_filename_discriminative_individual_root_optimal = f"{data_dir}/{experiment}/{input_dir}/discriminative_individual_root_optimal.csv"


        
        if os.path.exists(store_filename_optimal_checkpoint) and \
           os.path.exists(store_filename_optimal_auc) and \
           os.path.exists(store_filename_optimal_auc_per_rule) and \
           os.path.exists(store_filename_auc_average_per_rule) and \
           os.path.exists(store_filename_average) and \
           os.path.exists(store_filename_optimal_test_loss) and \
           (not args_current.memorization or os.path.exists(store_filename_near_freq_test_comparison)) and \
           (not args_current.memorization or os.path.exists(store_filename_string_average)) and \
           (not args_current.memorization or os.path.exists(store_filename_string_average_optimal)) and \
           (not args_current.memorization or os.path.exists(store_filename_string_average_optimal_auc)) and \
           (not args_current.memorization or args['incontext_input'] or os.path.exists(store_filename_memorization_results)) and \
           (not args_current.memorization or os.path.exists(store_filename_discriminative_individual)) and \
           (not args_current.memorization or os.path.exists(store_filename_discriminative_individual_optimal)) and \
           (not args_current.memorization or os.path.exists(store_filename_discriminative_individual_root)) and \
           (not args_current.memorization or os.path.exists(store_filename_discriminative_individual_root_optimal)) and \
           os.path.exists(store_filename_auc_average):
            continue


        # list of files that are not there
        for filename in [store_filename_optimal_checkpoint,
                         store_filename_optimal_auc,
                         store_filename_optimal_auc_per_rule,
                         store_filename_auc_average_per_rule,
                         store_filename_average,
                         store_filename_optimal_test_loss,
                         store_filename_auc_average,
                         store_filename_near_freq_test_comparison,
                         store_filename_string_average,
                         store_filename_string_average_optimal,
                         store_filename_string_average_optimal_auc,
                         store_filename_memorization_results,
                         store_filename_discriminative_individual,
                         store_filename_discriminative_individual_optimal,
                         store_filename_discriminative_individual_root,
                         store_filename_discriminative_individual_root_optimal]:
            if not os.path.exists(filename):
                print(f"missing file {filename.split('/')[-1]}")
                
                

        

        args['checkpoint_path_overwrite'] = None # so that the base tokenizer is called
        if args['model_name'] in model_revised_location:
            args['model_name'] = model_revised_location[args['model_name']]
        
        if "use_under_trained_tokens" not in args:
            args['use_under_trained_tokens'] = False


        if "add_instruction" not in args:
            args['add_instruction'] = False

        if "nlp_dataset" not in args:
            args['nlp_dataset'] = False
        
        if "memorization_algo" not in args:
            args['memorization_algo'] = "no_intervention"

        if "multilingual_grammar_name" not in args:
            args["multilingual_grammar_name"] = None

        if "multilingual" not in args:
            args['multilingual'] = False

        if "adaptive_training" not in args:
            args['adaptive_training'] = False
        
        if 'token_map_reverse' not in args:
            args['token_map_reverse'] = None

        if "include_grammar_rule_data" not in args:
            args['include_grammar_rule_data'] = False

        if args["multilingual"] and (args["multilingual_grammar_name"] is None):
            continue
        
        if args['model_name'] not in tokenizer_dict:
            try:
                tokenizer, checkpoint_path = get_tokenizer(Struct(**args), use_local_path=True)
            except:
                tokenizer, checkpoint_path = get_tokenizer(Struct(**args))
            tokenizer_dict[args['model_name']] = tokenizer
        
        tokenizer = tokenizer_dict[args['model_name']]

        if 'considered_incontext_repetitions' not in args:
            args['considered_incontext_repetitions'] = 1
        if 'counterfactual_memorization' not in args:
            args['counterfactual_memorization'] = False


        
        df = pd.read_csv(result_file, header=0)
        df = df[~df['label_id'].isin([-100, tokenizer_dict[args['model_name']].pad_token_id, tokenizer_dict[args['model_name']].bos_token_id, tokenizer_dict[args['model_name']].eos_token_id])]
        df['epoch'] = df['epoch'].fillna(0)

        print(f"Processing: {result_file}")

        if args['adaptive_training']:
            max_epoch_other_datasets = df[df['eval_dataset'] != 'train_sequences']['epoch'].max()
            assert df['epoch'].max() > max_epoch_other_datasets # 
            assert 'train_sequences' in df['eval_dataset'].unique()
            df_train_all_strings = df[
                (df['eval_dataset'] == 'train_sequences') &
                (df['epoch'] == df['epoch'].max())
            ].copy() # train sequences, but in the final epoch
            df_train_all_strings['epoch'] = max_epoch_other_datasets
            df = df[df['eval_dataset'] != 'train_sequences'].copy()
            df = pd.concat([df, df_train_all_strings], axis=0)    
        
        if not os.path.exists(store_filename_optimal_checkpoint):
            
            # there are some issues.
            try:
                df['target_token_negative_log_prob'] = df['target_token_negative_log_prob'].astype(float)
            except Exception as e:
                print(e)
                continue

            # optimal_checkpoint    
            optimal_epoch = df[df['eval_dataset'] == "test_sequences"].groupby(['epoch']).aggregate({'target_token_negative_log_prob': 'mean'})['target_token_negative_log_prob'].idxmin()
            df[df['epoch'] == optimal_epoch].to_csv(store_filename_optimal_checkpoint, index=False)



        if not os.path.exists(store_filename_average) or not os.path.exists(store_filename_optimal_test_loss):
            # average per epoch
            df_average = df.groupby(['eval_dataset', 'epoch']).aggregate({
                'target_token_negative_log_prob': 'mean',
                'correct': 'mean',
                'total_prob_mass': 'mean'
            }).reset_index()
            df_average.to_csv(store_filename_average, index=False)

            # optimal test loss epoch
            if 'test_sequences' in df_average['eval_dataset'].unique():
                optimal_test_loss_epoch = df_average[df_average['eval_dataset'] == "test_sequences"].groupby(['epoch']).aggregate({'target_token_negative_log_prob': 'mean'})['target_token_negative_log_prob'].idxmin()
                df_average[df_average['epoch'] == optimal_test_loss_epoch].to_csv(store_filename_optimal_test_loss, index=False)
            else:
                pd.DataFrame([]).to_csv(store_filename_optimal_test_loss, index=False)



        if args['incontext_input']:
            try:
                if args['nlp_dataset']:
                    df['length_input_tokens'] = df['length_input_tokens'] - len(tokenizer_dict[args['model_name']].encode(args['considered_incontext_examples'][1]))
                else:
                    df['length_input_tokens'] = df['length_input_tokens'] - len(args['considered_incontext_examples'][1])
            except:
                df['length_input_tokens'] = df['length_input_tokens'] - args['considered_incontext_examples'][1]
            
            

        if not os.path.exists(store_filename_optimal_auc) or not os.path.exists(store_filename_auc_average):
            assert 'test_sequences' in df['eval_dataset'].unique()
            if df['eval_dataset'].nunique() < 2 or args['adaptive_training']:
                pd.DataFrame([]).to_csv(store_filename_optimal_auc, index=False)
                pd.DataFrame([]).to_csv(store_filename_auc_average, index=False)
            else:


                # compute per-string loss. Get sample id
                df_list_temp = []        
                for key, df_item in df.groupby(['epoch', 'eval_dataset']):
                    df_item['sample_id'] = (df_item['length_input_tokens'] == 0).cumsum() - 1
                    df_list_temp.append(df_item)
                df = pd.concat(df_list_temp)

                print(df.groupby(['epoch', 'eval_dataset'])['sample_id'].max())
                result_inductive_bias_discriminative = []
                for epoch in df['epoch'].unique():

                    
                    df_working = df[(df['epoch'] == epoch)]

                    df_sample_average = df_working.groupby(['eval_dataset', 'sample_id']).aggregate({
                        'target_token_negative_log_prob': 'mean',
                        'correct': 'mean',      
                    }).reset_index() 
                    
                    
                    for dataset in df_sample_average['eval_dataset'].unique():
                            
                        if 'test_sequences' in dataset:
                            continue

                        print(dataset)
                        
                        xy = df_sample_average[df_sample_average['eval_dataset'].isin(['test_sequences', dataset])].copy()
                        num_base_dataset = xy[xy['eval_dataset'] == 'test_sequences'].shape[0]
                        num_compared_dataset = xy[xy['eval_dataset'] == dataset].shape[0]
                        

                        if num_compared_dataset < 2:
                            auc = None
                        else:



                            xy['label'] = xy.apply(lambda x: 1 if x['eval_dataset'] == 'test_sequences' else 0, axis=1)


                            seed = 5
                            xy = xy[['target_token_negative_log_prob', 'label']]
                            xy = xy.sample(frac=1, random_state=seed).reset_index(drop=True)
                            train, test = train_test_split(xy, 
                                                        stratify=xy['label'],
                                                        test_size=0.5, 
                                                        random_state=seed)


                            clf = LogisticRegression(random_state=0)
                            train = train.reset_index(drop=True)
                            test = test.reset_index(drop=True)
                            clf.fit(train[['target_token_negative_log_prob']], train['label'])
                            score = clf.score(test[['target_token_negative_log_prob']], test['label'])
                            probs = clf.predict_proba(test[['target_token_negative_log_prob']])[:, 1]
                            auc = roc_auc_score(test['label'], probs)
                            fpr, tpr, thresholds = roc_curve(test['label'], probs)


                        
                        result_inductive_bias_discriminative.append({
                            "epoch" : epoch,
                            "eval_dataset" : dataset,
                            "auc": auc
                        })


                        

                    
                result_inductive_bias_discriminative_df = pd.DataFrame(result_inductive_bias_discriminative)
                result_inductive_bias_discriminative_df.to_csv(store_filename_auc_average, index=False)

                
                # optimal auc epoch
                if 'non_grammatical_test_sequences_edit_distance_1' in result_inductive_bias_discriminative_df['eval_dataset'].unique():
                    optimal_auc_epoch = result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['eval_dataset'] == 'non_grammatical_test_sequences_edit_distance_1'].groupby(['epoch']).aggregate({'auc': 'mean'})['auc'].idxmax()
                    result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['epoch'] == optimal_auc_epoch].to_csv(store_filename_optimal_auc, index=False)
                else:
                    edit_1_datasets = [eval_dataset for eval_dataset in result_inductive_bias_discriminative_df['eval_dataset'].unique() if 'non_grammatical_test_sequences_edit_distance_1' in eval_dataset]
                    if len(edit_1_datasets) == 0:
                        pd.DataFrame([]).to_csv(store_filename_optimal_auc, index=False)
                    else:
                        # accumulate over all edit_1_datasets
                        result_inductive_bias_discriminative_df_accumulated = result_inductive_bias_discriminative_df[
                            result_inductive_bias_discriminative_df['eval_dataset'].isin(edit_1_datasets)
                        ].groupby(['epoch'])['auc'].sum().reset_index()
                        optimal_auc_epoch = result_inductive_bias_discriminative_df_accumulated.groupby(['epoch']).aggregate({'auc': 'mean'})['auc'].idxmax()
                        result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['epoch'] == optimal_auc_epoch].to_csv(store_filename_optimal_auc, index=False)


        if not os.path.exists(store_filename_optimal_auc_per_rule) or not os.path.exists(store_filename_auc_average_per_rule):
            eval_dataset_covered_by_rule = [eval_dataset for eval_dataset in df['eval_dataset'].unique() if 'covered_by_rule' in eval_dataset]
            
            if len(eval_dataset_covered_by_rule) == 0 or args['include_grammar_rule_data'] == False:
                pd.DataFrame([]).to_csv(store_filename_optimal_auc_per_rule, index=False)
                
            
            assert len(eval_dataset_covered_by_rule) % 2 == 0

            # compute per-string loss. Get sample id
            df_list_temp = []        
            for key, df_item in df.groupby(['epoch', 'eval_dataset']):
                df_item['sample_id'] = (df_item['length_input_tokens'] == 0).cumsum() - 1
                df_list_temp.append(df_item)
            df = pd.concat(df_list_temp)

            print(df.groupby(['epoch', 'eval_dataset'])['sample_id'].max())
            result_inductive_bias_discriminative = []
            for epoch in df['epoch'].unique():

                
                df_working = df[(df['epoch'] == epoch)]

                df_sample_average = df_working.groupby(['eval_dataset', 'sample_id']).aggregate({
                    'target_token_negative_log_prob': 'mean',
                    'correct': 'mean',      
                }).reset_index() 
                
                
                for dataset in eval_dataset_covered_by_rule:
                        
                    if 'covered_by_rule_perturbed' in dataset:
                        continue

                    rule_index = dataset.split("_")[-1]
                    dataset_perturbed = f"{'_'.join(dataset.split('_')[:-1])}_perturbed_{rule_index}"

                    xy = df_sample_average[df_sample_average['eval_dataset'].isin([dataset, dataset_perturbed])].copy()
                    num_base_dataset = xy[xy['eval_dataset'] == dataset].shape[0]
                    num_compared_dataset = xy[xy['eval_dataset'] == dataset_perturbed].shape[0]
                    

                    if num_compared_dataset < 2:
                        auc = None
                    else:



                        xy['label'] = xy.apply(lambda x: 1 if x['eval_dataset'] == dataset else 0, axis=1)


                        seed = 5
                        xy = xy[['target_token_negative_log_prob', 'label']]
                        xy = xy.sample(frac=1, random_state=seed).reset_index(drop=True)
                        train, test = train_test_split(xy, 
                                                    stratify=xy['label'],
                                                    test_size=0.5, 
                                                    random_state=seed)


                        clf = LogisticRegression(random_state=0)
                        train = train.reset_index(drop=True)
                        test = test.reset_index(drop=True)
                        clf.fit(train[['target_token_negative_log_prob']], train['label'])
                        score = clf.score(test[['target_token_negative_log_prob']], test['label'])
                        probs = clf.predict_proba(test[['target_token_negative_log_prob']])[:, 1]
                        auc = roc_auc_score(test['label'], probs)
                        fpr, tpr, thresholds = roc_curve(test['label'], probs)


                    
                    result_inductive_bias_discriminative.append({
                        "epoch" : epoch,
                        "eval_dataset" : dataset,
                        "auc": auc
                    })


                    

                
            result_inductive_bias_discriminative_df = pd.DataFrame(result_inductive_bias_discriminative)
            result_inductive_bias_discriminative_df.to_csv(store_filename_auc_average_per_rule, index=False)

            
            # # optimal auc epoch
            # if 'non_grammatical_test_sequences_edit_distance_1' in result_inductive_bias_discriminative_df['eval_dataset'].unique():
            #     optimal_auc_epoch = result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['eval_dataset'] == 'non_grammatical_test_sequences_edit_distance_1'].groupby(['epoch']).aggregate({'auc': 'mean'})['auc'].idxmax()
            #     result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['epoch'] == optimal_auc_epoch].to_csv(store_filename_optimal_auc, index=False)
            # else:
            #     edit_1_datasets = [eval_dataset for eval_dataset in result_inductive_bias_discriminative_df['eval_dataset'].unique() if 'non_grammatical_test_sequences_edit_distance_1' in eval_dataset]
            #     if len(edit_1_datasets) == 0:
            #         pd.DataFrame([]).to_csv(store_filename_optimal_auc, index=False)
            #     else:
            #         # accumulate over all edit_1_datasets
            #         result_inductive_bias_discriminative_df_accumulated = result_inductive_bias_discriminative_df[
            #             result_inductive_bias_discriminative_df['eval_dataset'].isin(edit_1_datasets)
            #         ].groupby(['epoch'])['auc'].sum().reset_index()
            #         optimal_auc_epoch = result_inductive_bias_discriminative_df_accumulated.groupby(['epoch']).aggregate({'auc': 'mean'})['auc'].idxmax()
            #         result_inductive_bias_discriminative_df[result_inductive_bias_discriminative_df['epoch'] == optimal_auc_epoch].to_csv(store_filename_optimal_auc, index=False)

        else:
            pd.DataFrame([]).to_csv(store_filename_optimal_auc_per_rule, index=False)


        if args_current.memorization and (not os.path.exists(store_filename_near_freq_test_comparison) or not os.path.exists(store_filename_string_average)):
        

            meta_data_filename = f"../data/{args['grammar_name'].replace('_counterfactual', '')}/meta_data_{args['grammar_name'].replace('_counterfactual', '')}_{args['num_samples']}_5.pkl"
            
            if os.path.exists(meta_data_filename):
                with open(meta_data_filename, 'rb') as f:
                    string_meta_data = pickle.load(f)
            else:
                string_meta_data = None           

            token_to_terminal_map = {}
            def convert_token_seq_to_terminal_seq(tokenizer, token_sequence):
                terminal_sequence = []
                for token in token_sequence:
                    if token not in token_to_terminal_map:
                        if args['token_map_reverse'] is None:
                            token_to_terminal_map[token] = tokenizer.decode(token)
                        else:
                            token_to_terminal_map[token] = args['token_map_reverse'][tokenizer.decode(token)]
                    terminal_sequence.append(token_to_terminal_map[token])
                return tuple(terminal_sequence)
            

            if args['incontext_input']:
                args['considered_incontext_examples'] = args['considered_incontext_examples'][0]

            # data_dict, _, _ = get_data(Struct(**args))
            df_list_temp = []

            selected_eval_datasets = [eval_dataset for eval_dataset in df['eval_dataset'].unique() if "non_grammatical" not in eval_dataset and "modified_tokens" not in eval_dataset and "test_sequences_" not in eval_dataset]

            
            cache_process_eval_dataset = {}
            token_to_gen_prob_map = None
            for key, df_item in df.groupby(['epoch', 'eval_dataset']):
                # if key[1] not in selected_eval_datasets:
                #     continue
                
                if key in cache_process_eval_dataset:
                    df_item['sample_id'] = cache_process_eval_dataset[key]['sample_id']
                    df_item['token_sequence'] = cache_process_eval_dataset[key]['token_sequence']
                    df_item['terminal_sequence'] = cache_process_eval_dataset[key]['terminal_sequence']
                    df_item['language_gen_prob'] = cache_process_eval_dataset[key]['language_gen_prob']
                    df_list_temp.append(df_item)
                    continue


                df_item['sample_id'] = (df_item['length_input_tokens'] == 0).cumsum() - 1
                
                print(key, df_item['sample_id'].max())
                # quit()
                
                
                # token sequence retrieve
                list_df_item = []
                for _, df_sample in df_item.groupby('sample_id'):
                    df_sample = df_sample.sort_values('length_input_tokens')
                    token_sequence = tuple(df_sample['label_id'].values)
                    df_sample['token_sequence'] = [token_sequence] * len(df_sample)
                    list_df_item.append(df_sample)
                df_item = pd.concat(list_df_item)
                try:
                    df_item['terminal_sequence'] = df_item['token_sequence'].apply(lambda x: convert_token_seq_to_terminal_seq(tokenizer_dict[args['model_name']], x))
                    if key[1] not in selected_eval_datasets or string_meta_data is None:
                        df_item['language_gen_prob'] = np.nan
                    else:
                        if (not args['incontext_input']) and (args['incontext_data_source'] is not None) and key[1] == "train_sequences": # ood experiment
                            df_item['language_gen_prob'] = np.nan
                        else:
                            df_item['language_gen_prob'] = df_item['terminal_sequence'].apply(lambda x: string_meta_data['sequence_prob_dict'][x])
                except Exception as e:
                    if token_to_gen_prob_map is None:
                        
                        if os.path.exists(f"{data_dir}/{experiment}/{input_dir}/token_to_gen_prob_map.pkl"):
                            with open(f"{data_dir}/{experiment}/{input_dir}/token_to_gen_prob_map.pkl", 'rb') as f:
                                token_to_gen_prob_map = pickle.load(f)
                                # token_to_gen_prob_map_modified = {}
                                # for token_seq in token_to_gen_prob_map:
                                #     token_to_gen_prob_map_modified[tuple(tensor(token_seq).tolist())] = token_to_gen_prob_map[token_seq]
                                # token_to_gen_prob_map = token_to_gen_prob_map_modified
                        else:
                            print(f"{data_dir}/{experiment}/{input_dir}/token_to_gen_prob_map.pkl does not exist")
                            raise RuntimeError()  

                    df_item['terminal_sequence'] = "Not retrieved"
                    if key[1] not in selected_eval_datasets or string_meta_data is None:
                        df_item['language_gen_prob'] = np.nan
                    else:
                        if (not args['incontext_input']) and (args['incontext_data_source'] is not None) and key[1] == "train_sequences": # ood experiment
                            df_item['language_gen_prob'] = np.nan
                        else:
                            df_item['language_gen_prob'] = df_item['token_sequence'].apply(lambda x: token_to_gen_prob_map[x])

                
                df_list_temp.append(df_item)

                cache_process_eval_dataset[key] = {
                    "sample_id": df_item['sample_id'].values,
                    "token_sequence": df_item['token_sequence'].values,
                    "terminal_sequence": df_item['terminal_sequence'].values,
                    "language_gen_prob": df_item['language_gen_prob'].values,
                }

            multi_index_columns = ['correct', 'target_token_negative_log_prob']    
            df_sample_average = pd.concat(df_list_temp).groupby(['epoch', 'sample_id', 'token_sequence', 'terminal_sequence', 'eval_dataset']).aggregate({
                'correct': ['mean', list],
                'target_token_negative_log_prob': ['mean', list],
                'language_gen_prob': 'median',
            }).reset_index()
            df_sample_average.columns = [col[0] if col[0] not in multi_index_columns else '_'.join(col) for col in df_sample_average.columns]
            df_sample_average = df_sample_average.rename(columns={
                'correct_mean': 'correct',
                'target_token_negative_log_prob_mean': 'target_token_negative_log_prob',
            })

            # afterwards
            dataset_specific_freq = {}
            total = 0
            
            # frequency of train sequences
            considered_epoch = df_sample_average[(df_sample_average['eval_dataset'].isin(['train_sequences']))]['epoch'].min()
            for i, row in df_sample_average[
                (df_sample_average['eval_dataset'].isin(['train_sequences'])) &
                (df_sample_average['epoch'] == considered_epoch)
            ].iterrows():
                token_sequence = row['token_sequence']
                if token_sequence not in dataset_specific_freq:
                    dataset_specific_freq[token_sequence] = 0
                dataset_specific_freq[token_sequence] += 1
                total += 1
            assert total > 0
            
            
            
            # # normalize frequency
            # for token_sequence in dataset_specific_freq:
            #     dataset_specific_freq[token_sequence] = dataset_specific_freq[token_sequence] / total
                        
            df_sample_average['dataset_gen_prob'] = df_sample_average.apply(
                lambda x: dataset_specific_freq[x['token_sequence']] if x['eval_dataset'] in ['train_sequences'] else None, axis=1
            )
            df_sample_average.to_csv(store_filename_string_average, index=False)

            print(df_sample_average)
            print(df_sample_average['eval_dataset'].unique())
            

            if string_meta_data is not None and (args['incontext_input'] or (args['incontext_data_source'] is None)):
                        
                

                list_nearest_test_comparison_result = []
                for eval_dataset in selected_eval_datasets:
                    if eval_dataset == "test_sequences":
                        continue
                    list_nearest_test_comparison_result.append(
                        compare_with_nearest_test_sequence(df_target=df_sample_average, training_dataset=eval_dataset, test_dataset='test_sequences')
                    )

                df_result_nearest_comparison = pd.concat(list_nearest_test_comparison_result)
                df_result_nearest_comparison.to_csv(store_filename_near_freq_test_comparison, index=False)
            else:
                print("Skipping: ", store_filename_near_freq_test_comparison)
                pd.DataFrame([]).to_csv(store_filename_near_freq_test_comparison, index=False)

        if args_current.memorization and (not os.path.exists(store_filename_string_average_optimal_auc)):
            df_sample_average = pd.read_csv(store_filename_string_average)
            try:
                df_optimal_auc = pd.read_csv(store_filename_optimal_auc)
                assert df_optimal_auc['epoch'].nunique() == 1
                optimal_epoch_auc = df_optimal_auc['epoch'].unique()[0]
                assert optimal_epoch_auc in df_sample_average['epoch'].unique()
                df_sample_average[df_sample_average['epoch'] == optimal_epoch_auc].to_csv(store_filename_string_average_optimal_auc, index=False)
            except:
                print("Skipping: ", store_filename_string_average_optimal_auc)
                pd.DataFrame([]).to_csv(store_filename_string_average_optimal_auc, index=False)

        
        if args_current.memorization and (not os.path.exists(store_filename_string_average_optimal)):
            df_sample_average = pd.read_csv(store_filename_string_average)
            try:
                df_optimal_test_loss = pd.read_csv(store_filename_optimal_test_loss)
                assert df_optimal_test_loss['epoch'].nunique() == 1
                optimal_epoch_test_loss = df_optimal_test_loss['epoch'].unique()[0]
                assert optimal_epoch_test_loss in df_sample_average['epoch'].unique()
                df_sample_average[df_sample_average['epoch'] == optimal_epoch_test_loss].to_csv(store_filename_string_average_optimal, index=False)
            except:
                print("Skipping: ", store_filename_string_average_optimal)
                pd.DataFrame([]).to_csv(store_filename_string_average_optimal, index=False)

        
        

        if args_current.memorization and (not os.path.exists(store_filename_discriminative_individual)) and (not args['adaptive_training']):
            df_string_average = pd.read_csv(store_filename_string_average)
            if "non_grammatical_test_sequences_edit_distance_1" not in df_string_average['eval_dataset'].unique():
                print("Skipping: ", store_filename_discriminative_individual)
                pd.DataFrame([]).to_csv(store_filename_discriminative_individual, index=False)
            else:
                
                print(df_string_average['eval_dataset'].unique())
                assert "test_sequences" in df_string_average['eval_dataset'].unique()
                assert "non_grammatical_test_sequences_edit_distance_1" in df_string_average['eval_dataset'].unique()
                df_string_average = df_string_average[
                    df_string_average['eval_dataset'].isin(['test_sequences', 'non_grammatical_test_sequences_edit_distance_1'])
                ]
                df_string_average['token_sequence'] = df_string_average['token_sequence'].apply(lambda x: literal_eval(x))
                res = get_edit_distance(df=df_string_average, eval_dataset_1='test_sequences', eval_dataset_2='non_grammatical_test_sequences_edit_distance_1')
                
                # process
                result_discriminative_indvidual = []
                for key, df_item in df_string_average.groupby(['epoch']):
                    df_item_test = df_item[df_item['eval_dataset'] == 'test_sequences'].copy()
                    df_item_incorrect = df_item[df_item['eval_dataset'] == 'non_grammatical_test_sequences_edit_distance_1'].copy()
                    df_item_test['min_distance'] = df_item_test['token_sequence'].apply(lambda x: res[x][0])
                    df_item_test['min_distant_incorrect_sample_ids'] = df_item_test['token_sequence'].apply(lambda x: res[x][1])
                    df_item_test['min_distant_target_token_negative_log_prob_list'] = df_item_test['min_distant_incorrect_sample_ids'].apply(lambda x: df_item_incorrect[df_item_incorrect['sample_id'].isin(x)]['target_token_negative_log_prob'].tolist())
                    df_item_test['min_distant_target_token_negative_log_prob'] = df_item_test['min_distant_target_token_negative_log_prob_list'].apply(lambda x: min(x))
                    df_item_test['min_distant_correct_list'] = df_item_test['min_distant_incorrect_sample_ids'].apply(lambda x: df_item_incorrect[df_item_incorrect['sample_id'].isin(x)]['correct'].tolist())
                    df_item_test['min_distant_correct'] = df_item_test['min_distant_correct_list'].apply(lambda x: max(x))
                    df_item_test['discrimination_target_token_negative_log_prob'] = df_item_test['min_distant_target_token_negative_log_prob'] > df_item_test['target_token_negative_log_prob']
                    df_item_test['discrimination_correct'] = df_item_test['min_distant_correct'] < df_item_test['correct']
                    
                    result_discriminative_indvidual.append(df_item_test)

                df_discriminative_indvidual = pd.concat(result_discriminative_indvidual)
                df_discriminative_indvidual.to_csv(store_filename_discriminative_individual, index=False)
                print("Discriminative results generated per individual")

            


        else:
            print("Skipping: ", store_filename_discriminative_individual)
            pd.DataFrame([]).to_csv(store_filename_discriminative_individual, index=False)

                

        
        if args_current.memorization and (not os.path.exists(store_filename_discriminative_individual_optimal)) and (not args['adaptive_training']):
            try:
                df_discriminative_indvidual = pd.read_csv(store_filename_discriminative_individual)
                df_optimal_auc = pd.read_csv(store_filename_optimal_auc)
                assert df_optimal_auc['epoch'].nunique() == 1
                optimal_epoch_auc = df_optimal_auc['epoch'].unique()[0]
                assert optimal_epoch_auc in df_discriminative_indvidual['epoch'].unique()
                df_discriminative_indvidual[df_discriminative_indvidual['epoch'] == optimal_epoch_auc].to_csv(store_filename_discriminative_individual_optimal, index=False)
            except:
                print("Skipping: ", store_filename_discriminative_individual_optimal)
                pd.DataFrame([]).to_csv(store_filename_discriminative_individual_optimal, index=False)
        else:
            print("Skipping: ", store_filename_discriminative_individual_optimal)
            pd.DataFrame([]).to_csv(store_filename_discriminative_individual_optimal, index=False)



        if args_current.memorization and (not os.path.exists(store_filename_discriminative_individual_root)) and (not args['adaptive_training']) and (not args['nlp_dataset']):
                # compare with nearest at root (during data generation)
                meta_data_lexer_edit_filename = f"../data/{args['grammar_name'].replace('_counterfactual', '')}/meta_data_lexer_edit_{args['grammar_name'].replace('_counterfactual', '')}_{args['num_samples']}_5.pkl"
                with open(meta_data_lexer_edit_filename, 'rb') as f:
                    meta_data_lexer_edit = pickle.load(f)

                    # combine
                    for key in ['non_terminal_applied_position_map', 'perturbation_result']:
                        if key not in meta_data_lexer_edit:
                            continue
                        temp = {}
                        for inner_key in meta_data_lexer_edit[key]:
                            for sequence in meta_data_lexer_edit[key][inner_key]:
                                temp[sequence] = meta_data_lexer_edit[key][inner_key][sequence]
                        meta_data_lexer_edit[key] = temp

                        
                    if 'sequence_to_perturbed_sequence_map' in meta_data_lexer_edit:
                        
                        df_string_average = pd.read_csv(store_filename_string_average)
                        df_string_average['terminal_sequence'] = df_string_average['terminal_sequence'].apply(literal_eval)
                        
                        
                        dict_unique_sequences_edit = {}
                        for eval_dataset in df_string_average['eval_dataset'].unique():
                            if "edit_distance" not in eval_dataset:
                                continue
                            print(eval_dataset)
                            for _, row in df_string_average[(df_string_average['eval_dataset'] == eval_dataset)].iterrows():
                                if row['terminal_sequence'] not in dict_unique_sequences_edit:
                                    dict_unique_sequences_edit[row['terminal_sequence']] = {}
                                if row['epoch'] not in dict_unique_sequences_edit[row['terminal_sequence']]:
                                    dict_unique_sequences_edit[row['terminal_sequence']][row['epoch']] = (row['eval_dataset'], 
                                                                                                            row['sample_id'], 
                                                                                                            row['terminal_sequence'], 
                                                                                                            row['target_token_negative_log_prob'],
                                                                                                            row['target_token_negative_log_prob_list'],
                                                                                                            tuple(position for position, _, _ in meta_data_lexer_edit['perturbation_result'][row['terminal_sequence']])
                                                                                                        )

                        df_string_average['nearest_edited_sequences'] = df_string_average.apply(
                            lambda row: [dict_unique_sequences_edit[edited_sequence][row['epoch']] for edited_sequence in dict_unique_sequences_edit
                                            if edited_sequence in meta_data_lexer_edit['sequence_to_perturbed_sequence_map'][row['terminal_sequence']]
                                        ] if row['eval_dataset'] == 'test_sequences' else [],
                            axis=1
                        )
                        df_string_average['found_edited_sequence'] = df_string_average.apply(
                            lambda row: len(row['nearest_edited_sequences']),
                            axis=1
                        )
                        # min loss
                        df_string_average['edited_sequence_min_loss'] = df_string_average.apply(
                            # lambda row: sum([row['nearest_edited_sequences'][i][3] for i in range(len(row['nearest_edited_sequences']))])/row['found_edited_sequence'] if row['found_edited_sequence'] > 0 else None, axis=1
                            lambda row: min([row['nearest_edited_sequences'][i][3] for i in range(len(row['nearest_edited_sequences']))]) if row['found_edited_sequence'] > 0 else None, axis=1
                        )

                        df_string_average['edited_sequence_loss_list'] = df_string_average.apply(
                            lambda row: [row['nearest_edited_sequences'][i][3] for i in range(len(row['nearest_edited_sequences']))] if row['found_edited_sequence'] > 0 else None, axis=1
                        )


                        df_string_average[df_string_average['found_edited_sequence'] > 0].to_csv(store_filename_discriminative_individual_root, index=False)

                    else:
                        print("Skipping: ", store_filename_discriminative_individual_root)
                        pd.DataFrame([]).to_csv(store_filename_discriminative_individual_root, index=False)
        else:
            print("Skipping: ", store_filename_discriminative_individual_root)
            pd.DataFrame([]).to_csv(store_filename_discriminative_individual_root, index=False)


        if args_current.memorization and (not os.path.exists(store_filename_discriminative_individual_root_optimal)) and (not args['adaptive_training']):
            try:
                df_discriminative_indvidual = pd.read_csv(store_filename_discriminative_individual_root)
                df_optimal_auc = pd.read_csv(store_filename_optimal_auc)
                assert df_optimal_auc['epoch'].nunique() == 1
                optimal_epoch_auc = df_optimal_auc['epoch'].unique()[0]
                assert optimal_epoch_auc in df_discriminative_indvidual['epoch'].unique()
                df_discriminative_indvidual[df_discriminative_indvidual['epoch'] == optimal_epoch_auc].to_csv(store_filename_discriminative_individual_root_optimal, index=False)
            except:
                print("Skipping: ", store_filename_discriminative_individual_root_optimal)
                pd.DataFrame([]).to_csv(store_filename_discriminative_individual_root_optimal, index=False)
        else:
            pd.DataFrame([]).to_csv(store_filename_discriminative_individual_root_optimal, index=False)
        


        if args_current.memorization and (not args['incontext_input']) and (not args['nlp_dataset']) and (not os.path.exists(store_filename_memorization_results)):
            

            if (not args['incontext_input']) and (args['incontext_data_source'] is not None): # ood case
                print("Skipping: ", store_filename_memorization_results)
                pd.DataFrame([]).to_csv(store_filename_memorization_results, index=False)
                

            if args['grammar_name'].endswith("multilingual"):
                print("Skipping: ", store_filename_memorization_results)
                pd.DataFrame([]).to_csv(store_filename_memorization_results, index=False)
                


            df_item_test_loss = pd.read_csv(store_filename_average)
            df_item_nearest_comparison = pd.read_csv(store_filename_near_freq_test_comparison)
            # min distance
            df_item_nearest_comparison['min_distant_test_prob'] = df_item_nearest_comparison.apply(
                    lambda x: list(set(literal_eval(x['min_distant_test_prob']))) if x['eval_dataset'] != "test_sequences" else None, axis=1 
            )
            df_item_nearest_comparison['min_distance'] = df_item_nearest_comparison.apply(
                lambda x: x['min_distant_test_prob'][0] if x['eval_dataset'] != "test_sequences" and len(x['min_distant_test_prob']) == 1 else (
                    "Wrong" if x['eval_dataset'] != "test_sequences" and len(x['min_distant_test_prob']) > 1 else 0
                ), axis=1
            )
            assert (df_item_nearest_comparison['min_distance'] == 'Wrong').sum() == 0
            df_item_nearest_comparison['min_distant_test_sample_ids'] = df_item_nearest_comparison.apply(
                    lambda x: list(set(literal_eval(x['min_distant_test_sample_ids']))) if x['eval_dataset'] != "test_sequences" else x['min_distant_test_sample_ids'], axis=1
            )

            eval_dataset = 'train_sequences'
            memorization_approach_goodname = {
                        'recollection_memorization': 'Recollection',
                        'adaptive_recollection_memorization': 'Adaptive-recollection',
                        'counterfactual_memorization': 'Counterfactual',
                        'contextual_memorization': 'Contextual',
            }
            optimal_test_performance_epoch = df_item_test_loss[df_item_test_loss['eval_dataset'] == "test_sequences"].groupby(['epoch']).aggregate({'target_token_negative_log_prob': 'mean'})['target_token_negative_log_prob'].idxmin()
            list_memorization_result = []
            for metric in ['target_token_negative_log_prob', 'correct']:
                print(metric)
                optimal_performance = df_item_test_loss[(df_item_test_loss['epoch'] == optimal_test_performance_epoch) & 
                                                            (df_item_test_loss['eval_dataset'] == "test_sequences")
                                                            ][metric].item()

                if metric == 'target_token_negative_log_prob':                    
                    memorization_threshold = optimal_performance / 2
                else:
                    memorization_threshold = (1 - optimal_performance) / 2 + optimal_performance
                                

                

                df_relevant = df_item_nearest_comparison[
                    (df_item_nearest_comparison['eval_dataset'] == eval_dataset) | 
                    (df_item_nearest_comparison['compared_to'] == eval_dataset)
                ]
                

                if df_relevant[df_relevant['eval_dataset'] == eval_dataset].shape[0] == 0:
                    continue

                
                for approach in memorization_approach_goodname.keys():
                    # dropped_samples = 0
                    for (sample_id,), df_item in df_relevant.groupby(['sample_id']):
                        if "train_sequences" not in df_item['eval_dataset'].unique() or  "test_sequences" not in df_item['eval_dataset'].unique():
                                continue


                        if approach in ['counterfactual_memorization', 'contextual_memorization']:
                            assert df_item[df_item['eval_dataset'] == eval_dataset]['min_distance'].nunique() == 1
                            # if df_item[df_item['eval_dataset'] == eval_dataset]['min_distance'].unique()[0] != 0:
                            #     dropped_samples += 1
                            #     continue

                        # find best of test dataset
                        if args['adaptive_training']:
                            df_item_train = df_item[df_item['eval_dataset'] == 'train_sequences'].copy()
                            df_item_test = df_item[df_item['eval_dataset'] == 'test_sequences'].copy()
                            assert df_item_train.shape[0] == 1

                            if metric == "correct":
                                df_item_test_best_row = df_item_test[df_item_test['correct'] == df_item_test['correct'].max()].iloc[:1].copy()
                            elif metric == "target_token_negative_log_prob":
                                df_item_test_best_row = df_item_test[df_item_test['target_token_negative_log_prob'] == df_item_test['target_token_negative_log_prob'].min()].iloc[:1].copy()

                            df_item_test_best_row['epoch'] = df_item_train['epoch'].item()
                            df_item = pd.concat([df_item_train, df_item_test_best_row], axis=0)
 
                        
                        computed_memorization, start_of_memorization, best_contextual_recollection, memorization_has_started = get_start_of_memorization(
                                df=df_item, 
                                eval_dataset='test_sequences',
                                training_dataset="train_sequences",
                                metric=metric,
                                approach=approach.replace("adaptive_", ""),
                                memorization_threshold=memorization_threshold if "adaptive" in approach else None
                        )
                        computed_memorization['memorization'] = computed_memorization['memorization'].fillna(0)
                        computed_memorization['memorization_binary'] = computed_memorization['memorization'] > 0 # make memorization binary
                        computed_memorization['sample_id'] = sample_id
                        computed_memorization['approach'] = approach
                        computed_memorization['language_gen_prob'] = df_item[(df_item['eval_dataset'] == eval_dataset) & (df_item['epoch'] == df_item['epoch'].max())]['language_gen_prob'].item()
                        computed_memorization['dataset_gen_prob'] = df_item[(df_item['eval_dataset'] == eval_dataset) & (df_item['epoch'] == df_item['epoch'].max())]['dataset_gen_prob'].item()
                        computed_memorization['parameter'] = best_contextual_recollection
                        computed_memorization['metric'] = metric
                        computed_memorization['optimal_learning'] = optimal_performance
                        computed_memorization['distance_from_test'] = df_item[df_item['eval_dataset'] == eval_dataset]['min_distance'].unique()[0]
                        

                        list_memorization_result.append(computed_memorization)


                    
                    # print(f"dropped_samples: {dropped_samples}")
                    

            df_memorization_result = pd.concat(list_memorization_result)
            df_memorization_result['eval_dataset'] = eval_dataset
            df_memorization_result['optimal_epoch'] = optimal_test_performance_epoch
            df_memorization_result.to_csv(store_filename_memorization_results, index=False)

        else:
            pd.DataFrame([]).to_csv(store_filename_memorization_results, index=False)