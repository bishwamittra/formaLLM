#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 4 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-04,sws-8a100-02




#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi







time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_00_52_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_07_50_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_17_47_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_13_34_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_17_14_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_20_43_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_00_52_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_07_50_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_17_47_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_13_34_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_17_14_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_20_43_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_00_52_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_07_50_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_17_47_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_13_34_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_17_14_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_20_43_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_00_52_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_07_50_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_17_47_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_13_34_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_17_14_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_20_43_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_00_52_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_16_07_50_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_17_47_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_13_34_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_17_14_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name mistralai/Mistral-7B-v0.3 \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_17_20_43_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name google/gemma-2-2b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_12_22_59_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_01_56_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_16/checkpoint-13 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_04_44_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_16/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_11_07_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_13_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_12_22_59_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_01_56_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_16/checkpoint-13 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_04_44_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_16/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_11_07_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_13_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_12_22_59_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_01_56_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_16/checkpoint-13 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_04_44_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_16/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_11_07_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_13_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_12_22_59_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_01_56_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_16/checkpoint-13 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_04_44_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_16/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_11_07_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_13_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_12_22_59_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_16/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_01_56_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_16/checkpoint-13 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_04_44_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_16/checkpoint-12 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_11_07_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_64/checkpoint-20 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name EleutherAI/pythia-6.9b \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_13_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_64/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_12_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-64 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_19_09__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-80 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_16_00__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_23_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-22 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_20_22__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_01_46__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_20_25__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_10_01__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_15_13__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_10_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-10 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_07_27__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_04_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_12_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-64 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_19_09__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-80 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_16_00__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_23_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-22 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_20_22__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_01_46__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_20_25__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_10_01__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_15_13__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_10_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-10 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_07_27__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_04_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_12_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-64 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_19_09__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-80 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_16_00__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_23_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-22 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_20_22__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_01_46__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_20_25__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_10_01__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_15_13__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_10_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-10 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_07_27__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_04_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_12_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-64 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_19_09__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-80 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_16_00__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_23_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-22 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_20_22__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_01_46__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_20_25__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_10_01__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_15_13__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_10_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-10 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_07_27__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_04_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_12_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-64 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_19_09__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-80 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_16_00__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-48 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_23_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-22 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_20_22__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_14_01_46__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-18 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_20_25__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_10_01__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-192 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_15_15_13__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-224 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_10_02__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_11_save_checkpoint/checkpoint-10 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_07_27__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name /NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_13_04_51__NS_llm-1_nobackup_vnanda_llm_base_models_Llama-2-7b-hf_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_11_save_checkpoint/checkpoint-9 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_08_31_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_50_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_07_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_22_37_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_58_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-280 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_00_21_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_05_49_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_02_56_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_08_43_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_15_05_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-38 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_13_44_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-100 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_25_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-96 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_one_rule_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G101_test_G101_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_08_31_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_50_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_07_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_22_37_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_58_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-280 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_00_21_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_05_49_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_02_56_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_08_43_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_15_05_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-38 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_13_44_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-100 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_25_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-96 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_two_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G102_test_G102_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_08_31_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_50_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_07_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_22_37_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_58_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-280 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_00_21_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_05_49_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_02_56_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_08_43_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_15_05_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-38 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_13_44_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-100 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_25_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-96 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_three_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G103_test_G103_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_08_31_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_50_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_07_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_22_37_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_58_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-280 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_00_21_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_05_49_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_02_56_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_08_43_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_15_05_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-38 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_13_44_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-100 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_25_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-96 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_four_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G104_test_G104_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_08_31_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_50_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-50 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_07_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-14 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_16 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_22_37_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_58_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-280 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_00_21_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_256 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_05_49_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_02_56_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-256 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_08_43_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_1024 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_15_05_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-38 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 2 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_13_44_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-100 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 1 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --inference_only_mode \
                    --model_name meta-llama/Llama-3.2-1B \
                    --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_25_meta-llama_Llama-3.2-1B_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-96 \
                    --grammar_name pcfg_cfg3b_disjoint_terminals_five_rules_different  \
                    --num_samples 10000 \
                    --store_result \
                    --incontext_input \
                    --considered_training_samples 0  \
                    --considered_incontext_examples 0 \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128  \
                    --data_seed 5 \
                    --run_seed 0 \
                    --batch_size 4  \
                    --comment 2025_01_24_edit_distance_results_ft_G1_icl_G105_test_G105_64 \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance \
                    
