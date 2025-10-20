#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 4 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-01,sws-8a100-02




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
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_16 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_16 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G80_test_G80_16 \
                

                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_11_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_64/checkpoint-18 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_14_18_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_64/checkpoint-14 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_12_58_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_64/checkpoint-16 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_64 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_09_06_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_16/checkpoint-8 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_16 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_07_44_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_16/checkpoint-8 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_16 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_10_22_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_16/checkpoint-9 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_leaf_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 64  \
                --considered_incontext_examples 64 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G90_test_G90_16 \
                
