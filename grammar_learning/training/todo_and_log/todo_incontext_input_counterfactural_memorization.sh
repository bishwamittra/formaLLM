#!/bin/bash
#
#SBATCH --partition=h200       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h200grid-06,sws-8h200grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi




comment="2025_05_11_fixed_parameters_counterfactural_memorization_incontext_input_multiple_models"



for run_seed in "0" "1" "2"
do

    for considered_incontext_examples in "0"
    do

        
        # for grammar_name in  "pcfg_cfg3b_eq_len_skewed_prob_counterfactual" "pcfg_cfg3b_eq_len_skewed_prob_0.90_counterfactual" "pcfg_cfg3b_eq_len_uniform_prob_counterfactual"
        for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_all_rules_skewed_prob_counterfactual" "pcfg_cfg3b_eq_len_skewed_prob_counterfactual" "preg_alphabet_combined_counterfactual"
        do




            for considered_training_samples in "16" "64" "256" "1024"
            do

                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-1b" \
                        # "google/gemma-2-9b" \
                        # "EleutherAI/pythia-6.9b" \
                        # "EleutherAI/pythia-1b" \

                do

                    # excluding the string
                    time  TRANSFORMERS_VERBOSITY=error python training.py \
                    --inference_only_mode \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 1 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --incontext_input \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_incontext_examples ${considered_incontext_examples} \
                    --considered_eval_samples 128 \
                    --store_result \
                    --batch_size 1 \
                    --include_incorrect_random_eval \
                    --include_edit_distance_eval \
                    --combine_edit_distance \
                    


                    
                    # including the string
                    for counterfactual_string_index in "0" "1" "2"
                    do

                        
                        time TRANSFORMERS_VERBOSITY=error python training.py \
                        --inference_only_mode \
                        --model_name ${model_name} \
                        --grammar_name ${grammar_name} \
                        --num_samples 10000 \
                        --num_train_epochs 1 \
                        --data_seed 5 \
                        --run_seed ${run_seed} \
                        --comment ${comment} \
                        --incontext_input \
                        --considered_training_samples ${considered_training_samples} \
                        --considered_incontext_examples ${considered_incontext_examples} \
                        --considered_eval_samples 128 \
                        --store_result \
                        --batch_size 1 \
                        --include_incorrect_random_eval \
                        --include_edit_distance_eval \
                        --combine_edit_distance \
                        --counterfactual_memorization \
                        --counterfactual_string_index ${counterfactual_string_index} \
                        
                        
                        

                    done



                    
                    
                    

                done
            done
        done
    done
done











# for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_skewed_prob_0.75"
# do
#     for considered_incontext_examples in "0"
#     do
        
#         for considered_training_samples in "16" "64" "256" "1024"
#         do
#             for run_seed in "0" "1" "2"
#             do
#                 for model_name in "google/gemma-2-9b" \
#                                   "mistralai/Mistral-7B-v0.3" \
#                                   "EleutherAI/pythia-6.9b" \

                                
#                 do
#                     time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
#                     training.py \
#                     --inference_only_mode \
#                     --model_name ${model_name} \
#                     --grammar_name ${grammar_name} \
#                     --num_samples 10000 \
#                     --store_result \
#                     --incontext_input \
#                     --considered_training_samples ${considered_training_samples} \
#                     --considered_incontext_examples ${considered_incontext_examples} \
#                     --num_train_epochs 1 \
#                     --considered_eval_samples 128 \
#                     --data_seed 5 \
#                     --run_seed ${run_seed} \
#                     --batch_size 1 \
#                     --comment ${comment} \
#                     --include_edit_distance_eval \
#                     --include_incorrect_random_eval \
#                     --combine_edit_distance



#                 done
#             done
#         done
#     done
# done



