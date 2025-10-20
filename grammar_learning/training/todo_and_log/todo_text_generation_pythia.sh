#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-3a100grid-01,sws-8a100-03,sws-8a100-01,sws-3a40grid-01,sws-2a40grid-01,sws-2a40grid-02

# sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=2

comment="Week_2024.09.11_text_generation_callback"

for grammar_name in "pcfg_cfg3b_eq_len"
do
    for considered_training_samples in "32"
    do
        time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
        training.py \
        --model_name pythia-1b \
        --grammar_name ${grammar_name} \
        --num_samples 10000 \
        --num_train_epochs 100 \
        --run_seed 0 \
        --data_seed 5 \
        --batch_size 8 \
        --comment ${comment} \
        --considered_training_samples ${considered_training_samples} \
        --considered_eval_samples ${considered_training_samples} \
        --store_result \
        --generate_text \
        --learning_rate 0.000005 \
        --max_new_tokens 1 \
        --compute_msp \
        # --save_checkpoint \
        # --include_edit_distance_eval \
        # --include_incorrect_random_eval \
        # --combine_edit_distance \        

    done
done

