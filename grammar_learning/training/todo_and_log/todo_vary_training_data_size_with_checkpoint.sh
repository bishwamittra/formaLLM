#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4           # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-04,sws-8h100grid-05,sws-8h100grid-07,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi

comment="2025_01_24_save_checkpoint"



nvidia-smi
for model_name in "EleutherAI/pythia-6.9b" \
                    

do
    for run_seed in "0" "1" "2"
    do

        for considered_training_samples in "16" "64"
        do
                        
            for grammar_name in "pcfg_cfg3b_disjoint_terminals_one_rule_different" "pcfg_cfg3b_disjoint_terminals_two_rules_different" "pcfg_cfg3b_disjoint_terminals_three_rules_different"
            do
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
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
                --use_deepspeed \
                --save_best_model \
                  
            
            done
        done 
    done
done




