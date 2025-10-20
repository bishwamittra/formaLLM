#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 0-12:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


# sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=2


nvidia-smi


comment="2024_09_16_rerun_previous_grammars"
# for grammar_name in "pcfg_reverse_string" "pcfg_balanced_parenthesis"
# for grammar_name in "pcfg_cfg3b_disjoint_terminals"  "pcfg_4_3_1_2_3_4_5_6_7_8_9"
for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
do
    # for model_name in "gemma-7b" "gemma-2b" "mistral-7b" "llama2-7b"
    for model_name in "gpt2-large"
    do
        for considered_training_samples in "32"
        # for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
        do
            # pretrain model
            time torchrun \
            --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
            training.py \
            --use_deepspeed \
            --model_name ${model_name} \
            --grammar_name ${grammar_name} \
            --num_samples 50000 \
            --num_train_epochs 50 \
            --given_seed 5 \
            --batch_size 8 \
            --comment ${comment} \
            --considered_training_samples ${considered_training_samples} \
            --store_result \
            --learning_rate 0.00005 \
            --include_incorrect_random_eval \
            --include_edit_distance_eval \
            --include_grammar_edit_eval \
            --combine_edit_distance

            
        done 
    done
done
