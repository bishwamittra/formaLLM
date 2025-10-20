#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 5-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


# sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=2


comment="Week_2024.09.25_multiple_run_pythia"

for considered_training_samples in "32"
do
    # for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob"
    for grammar_name in "pcfg_cfg3b_disjoint_terminals"
    do
        for run_seed in "0" "1" "5"
        do
            # pretrain model
            time torchrun \
            --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
            training.py \
            --model_name gpt2-large \
            --grammar_name ${grammar_name} \
            --num_samples 10000 \
            --num_train_epochs 2 \
            --data_seed 5 \
            --considered_eval_samples 1000 \
            --batch_size 8 \
            --run_seed ${run_seed} \
            --comment ${comment} \
            --considered_training_samples ${considered_training_samples} \
            --store_result \
            --learning_rate 0.000005 \
            --include_edit_distance_eval \
            --include_incorrect_random_eval \
            --combine_edit_distance \
            # --exclude_test_data \
            # --generate_text \
            # --save_checkpoint \
            


        done
    done
done
