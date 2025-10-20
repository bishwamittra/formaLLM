#!/bin/bash
#
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-07,sws-8h100grid-05,sws-8h100grid-04



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
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_23_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_15_19_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_1024/checkpoint-160 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_14_08_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_06_43_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_12_13_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_17_02_57_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name EleutherAI/pythia-6.9b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_12_13_11_31_EleutherAI_pythia-6.9b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_12_13_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_10_37_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_1024/checkpoint-256 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_20_22_12_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_1024/checkpoint-288 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_19_22_59_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_14_51_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_1_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_09_42_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_0_2024_10_17_save_checkpoint_256/checkpoint-56 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name mistralai/Mistral-7B-v0.3 \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_10_18_20_02_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_disjoint_terminals_10000_2_2024_10_17_save_checkpoint_256/checkpoint-40 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.55  \
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
                --comment 2025_01_24_ft_G1_icl_G1955_test_G1955_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.60  \
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
                --comment 2025_01_24_ft_G1_icl_G1960_test_G1960_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.70  \
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
                --comment 2025_01_24_ft_G1_icl_G1970_test_G1970_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.80  \
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
                --comment 2025_01_24_ft_G1_icl_G1980_test_G1980_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 0  \
                --considered_incontext_examples 0 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 1  \
                --considered_incontext_examples 1 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 2  \
                --considered_incontext_examples 2 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 4  \
                --considered_incontext_examples 4 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 8  \
                --considered_incontext_examples 8 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 16  \
                --considered_incontext_examples 16 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 0 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 1 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples 32  \
                --considered_incontext_examples 32 \
                --num_train_epochs 1 \
                --considered_eval_samples 1024 \
                --data_seed 5 \
                --run_seed 2 \
                --batch_size 4  \
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_07_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_1024/checkpoint-224 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_01_15_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_1024/checkpoint-320 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_10_04_09_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_1024/checkpoint-928 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_1024 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_20_01_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_0_2024_11_09_save_checkpoint_256/checkpoint-48 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_21_39_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_1_2024_11_09_save_checkpoint_256/checkpoint-88 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --inference_only_mode \
                --model_name google/gemma-2-2b \
                --checkpoint_path_overwrite results/revised/generalization/fine-tuning/with_checkpoints/output_2024_11_09_23_19_google_gemma-2-2b_pcfg_cfg3b_disjoint_terminals_10000_2_2024_11_09_save_checkpoint_256/checkpoint-168 \
                --grammar_name pcfg_cfg3b_disjoint_terminals_all_rules_0.90  \
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
                --comment 2025_01_24_ft_G1_icl_G1990_test_G1990_256 \
                
