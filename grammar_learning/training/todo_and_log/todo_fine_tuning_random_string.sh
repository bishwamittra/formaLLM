#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:1            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=200GB             # Memory pool for all cores (see also --mem-per-cpu)
# SBATCH --exclude=sws-8h100grid-07


# sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=1
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi



# comment="2025_08_13_memorization_random_string_Till_fine_tuning"

# for considered_training_samples in "1"
# do
#     for grammar_name in "random_string_alphabet_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_length_1024_only_training_string" \
#                         "random_string_alphabet_a_b_length_1024_only_training_string"  \
#                         "random_string_first_half_high_entropy_second_half_low_entropy" \
#                         "random_string_first_half_low_entropy_second_half_high_entropy"

#     do
#         for run_seed in "0"
#         do

#             for model_name in "EleutherAI/pythia-1b" \
#                             # "Qwen/Qwen2.5-7B" \
                            
                                                         
                    

#             do
#                 time torchrun \
#                 --nproc_per_node=$GPUS_PER_NODE \
#                 training.py \
#                 --model_name ${model_name} \
#                 --grammar_name ${grammar_name} \
#                 --num_samples 10000 \
#                 --num_train_epochs 50 \
#                 --data_seed 5 \
#                 --run_seed ${run_seed} \
#                 --comment ${comment} \
#                 --considered_training_samples ${considered_training_samples} \
#                 --considered_eval_samples 1024 \
#                 --store_result \
#                 --include_incorrect_random_eval \
#                 --include_edit_distance_eval \
#                 # --use_deepspeed \


#             done
#         done
#     done
# done




# comment="2025_08_14_sensitive_vs_nonsensitive_tokens_fine_tuning"

# for considered_training_samples in "1" "4" "16" "64"
# do
#     # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity"
#     for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_3"

#     do
#         for run_seed in "0"
#         do

#             for model_name in "EleutherAI/pythia-1b" \
#                             # "Qwen/Qwen2.5-7B" \
                            
                                                         
                    

#             do
#                 time torchrun \
#                 --nproc_per_node=$GPUS_PER_NODE \
#                 training.py \
#                 --model_name ${model_name} \
#                 --grammar_name ${grammar_name} \
#                 --num_samples 10000 \
#                 --num_train_epochs 50 \
#                 --data_seed 5 \
#                 --run_seed ${run_seed} \
#                 --comment ${comment} \
#                 --considered_training_samples ${considered_training_samples} \
#                 --considered_eval_samples 128 \
#                 --store_result \
#                 --include_incorrect_random_eval \
#                 --include_edit_distance_eval \
#                 # --use_deepspeed \


#             done
#         done
#     done
# done



comment="2025_08_14_sensitive_vs_nonsensitive_tokens_fine_tuning"

for considered_training_samples in "12"
do
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity"
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_3"
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_3_deduplicated"
    for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_6_deduplicated" \
                        "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_7_deduplicated"

    do
        for run_seed in "0"
        do

            for model_name in "EleutherAI/pythia-1b" \
                            # "Qwen/Qwen2.5-7B" \
                            
                                                         
                    

            do
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
                training.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 100 \
                --data_seed 5 \
                --run_seed ${run_seed} \
                --comment ${comment} \
                --considered_training_samples ${considered_training_samples} \
                --considered_eval_samples 128 \
                --store_result \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                # --use_deepspeed \


            done
        done
    done
done



