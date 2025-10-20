#!/bin/bash
#
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres=gpu:1          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=244GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=1
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir



comment="2025_09_16_memorization_based_intervention_fine_tuning"




for considered_training_samples in "16"
# for considered_training_samples in "16" "64"
# for considered_training_samples in "256" "1024"
do


    for run_seed in "0" "1" "2"
    # for run_seed in "0"
    do


        # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob_0.75" 
        # for grammar_name in ""pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob""
        for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg_extended_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_uniform_prob" "pcfg_cfg3b_eq_len_uniform_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob" "pcfg_cfg3b_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg_extended_eq_len_skewed_prob" "pcfg_cfg3b_disjoint_terminals_skewed_prob"
        do


            for model_name in \
                    "EleutherAI/pythia-1b" \
                    # "mistralai/Mistral-7B-v0.3" \
                    # Qwen/Qwen2.5-7B \
                    # "meta-llama/Llama-3.2-1B" \
                    # "google/gemma-2-2b" \
                    

            
            do

                for memorization_algo in \
                                         "no_intervention"  \
                                         "deduplication" \
                                        # "remove_before_memorized_and_never_put_back" \
                                        #  "remove_before_memorized" \
                                        #  "training_variance" \
                                        #  "remove_after_memorized" \
                                         
                       

                do


                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 50 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_eval_samples 1024 \
                    --store_result \
                    --memorization_algo ${memorization_algo} \
                    # --use_deepspeed \
                    # --include_incorrect_random_eval \
                    # --include_edit_distance_eval \
                    # --combine_edit_distance \



                done
            done
        done
    done
done




comment="2025_09_06_memorization_mitigation_test_ood_fine_tuning"











for run_seed in "0" "1" "2"
do

    for considered_training_samples in "256"
    # for considered_training_samples in "16" "64" "256" "1024"
    # for considered_training_samples in "16" "64" "256" "1024"
    do


        for training_grammar_name in "pcfg_cfg3b_eq_len_uniform_prob"
        # for training_grammar_name in "pcfg_cfg3b_disjoint_terminals"
        do


            for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
            # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
            do
            

                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-1b" \
                        # "google/gemma-2-2b" \

                do

                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 50 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --incontext_data_source ${training_grammar_name}:train_sequences \
                    --considered_eval_samples 1024 \
                    --store_result \
                    # --use_deepspeed \
                    # --include_incorrect_random_eval \
                    # --include_edit_distance_eval \
                    # --combine_edit_distance \
                    # --include_grammar_edit_eval \
                    

                done
            done
        done
    done
done









for run_seed in "0" "1" "2"
do

    for considered_training_samples in "256"
    # for considered_training_samples in "16" "64" "256" "1024"
    # for considered_training_samples in "16" "64" "256" "1024"
    do


        # for training_grammar_name in "pcfg_cfg3b_eq_len_uniform_prob"
        for training_grammar_name in "pcfg_cfg3b_disjoint_terminals"
        do


            # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
            for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
            do
            

                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-1b" \
                        # "google/gemma-2-2b" \

                do

                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 50 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --incontext_data_source ${training_grammar_name}:train_sequences \
                    --considered_eval_samples 1024 \
                    --store_result \
                    # --use_deepspeed \
                    # --include_incorrect_random_eval \
                    # --include_edit_distance_eval \
                    # --combine_edit_distance \
                    # --include_grammar_edit_eval \
                    

                done
            done
        done
    done
done
