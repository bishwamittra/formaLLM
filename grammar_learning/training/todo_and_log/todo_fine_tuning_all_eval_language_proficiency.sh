#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-01


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=2
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir



nvidia-smi



comment="2025_09_14_all_eval_language_proficiency_fine_tuning"


for run_seed in "0" "1" "2"
# for run_seed in "0"
do

    # for considered_training_samples in "16"
    for considered_training_samples in "16" "64" "256" "1024"
    do


        for grammar_name in "pcfg_cfg3b_disjoint_terminals" 
        do

            for model_name in \
                    "EleutherAI/pythia-1b" \
                    # "google/gemma-2-2b" \
                    # "mistralai/Mistral-7B-v0.3" \

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
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --include_grammar_edit_eval \
                --combine_edit_distance \


            done
        done
    done
done



