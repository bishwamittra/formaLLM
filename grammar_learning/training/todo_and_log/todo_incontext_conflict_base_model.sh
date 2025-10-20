#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 4 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-03,sws-8a100-02




#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi









    comment=f"Week_2025.02_05_edit_distance_results_ft_G1_icl_G1_test_G101_0"

    
    for considered_incontext_examples in "0" "16" "64" "256" "1024"                    
    do

    
        for grammar_name in pcfg_cfg3b_disjoint_terminals_one_rule_different    
        do

        
            for run_seed in "0" "1" "2"
            do

            
                for model_name in "mistralai/Mistral-7B-v0.3"                                 "EleutherAI/pythia-6.9b"                                 "google/gemma-2-2b"                                 "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf"                                 "meta-llama/Llama-3.2-1B"                               

                do
                    time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID                     training.py                     --inference_only_mode                     --model_name ${model_name}                     --grammar_name ${grammar_name}                     --num_samples 10000                     --store_result                     --incontext_input                     --considered_training_samples ${considered_incontext_examples}                     --considered_incontext_examples ${considered_incontext_examples}                     --num_train_epochs 1                     --considered_eval_samples 128                     --data_seed 5                     --run_seed ${run_seed}                     --batch_size 1                     --incontext_data_source ../data/pcfg_cfg3b_disjoint_terminals/sequences_w_edit_distance_pcfg_cfg3b_disjoint_terminals_10000_5.pkl:train_sequences                     --comment ${comment}                     --include_edit_distance_eval                     --include_incorrect_random_eval                     --combine_edit_distance
                    

                done
            done
        done
    done


    


    comment=f"Week_2025.02_05_edit_distance_results_ft_G1_icl_G1_test_G102_0"

    
    for considered_incontext_examples in "0" "16" "64" "256" "1024"                    
    do

    
        for grammar_name in pcfg_cfg3b_disjoint_terminals_two_rules_different    
        do

        
            for run_seed in "0" "1" "2"
            do

            
                for model_name in "mistralai/Mistral-7B-v0.3"                                 "EleutherAI/pythia-6.9b"                                 "google/gemma-2-2b"                                 "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf"                                 "meta-llama/Llama-3.2-1B"                               

                do
                    time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID                     training.py                     --inference_only_mode                     --model_name ${model_name}                     --grammar_name ${grammar_name}                     --num_samples 10000                     --store_result                     --incontext_input                     --considered_training_samples ${considered_incontext_examples}                     --considered_incontext_examples ${considered_incontext_examples}                     --num_train_epochs 1                     --considered_eval_samples 128                     --data_seed 5                     --run_seed ${run_seed}                     --batch_size 1                     --incontext_data_source ../data/pcfg_cfg3b_disjoint_terminals/sequences_w_edit_distance_pcfg_cfg3b_disjoint_terminals_10000_5.pkl:train_sequences                     --comment ${comment}                     --include_edit_distance_eval                     --include_incorrect_random_eval                     --combine_edit_distance
                    

                done
            done
        done
    done


    


    comment=f"Week_2025.02_05_edit_distance_results_ft_G1_icl_G1_test_G103_0"

    
    for considered_incontext_examples in "0" "16" "64" "256" "1024"                    
    do

    
        for grammar_name in pcfg_cfg3b_disjoint_terminals_three_rules_different    
        do

        
            for run_seed in "0" "1" "2"
            do

            
                for model_name in "mistralai/Mistral-7B-v0.3"                                 "EleutherAI/pythia-6.9b"                                 "google/gemma-2-2b"                                 "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf"                                 "meta-llama/Llama-3.2-1B"                               

                do
                    time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID                     training.py                     --inference_only_mode                     --model_name ${model_name}                     --grammar_name ${grammar_name}                     --num_samples 10000                     --store_result                     --incontext_input                     --considered_training_samples ${considered_incontext_examples}                     --considered_incontext_examples ${considered_incontext_examples}                     --num_train_epochs 1                     --considered_eval_samples 128                     --data_seed 5                     --run_seed ${run_seed}                     --batch_size 1                     --incontext_data_source ../data/pcfg_cfg3b_disjoint_terminals/sequences_w_edit_distance_pcfg_cfg3b_disjoint_terminals_10000_5.pkl:train_sequences                     --comment ${comment}                     --include_edit_distance_eval                     --include_incorrect_random_eval                     --combine_edit_distance
                    

                done
            done
        done
    done