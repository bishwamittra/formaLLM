#!/bin/bash
#
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres=gpu:4            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-07


# sws-8a100-03,sws-8a100-01,sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.out      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=1
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="../training"
cd $workdir
nvidia-smi



comment="2025_07_11_nlp_dataset_fine_tuning"

for considered_training_samples in "1024"
do
    for grammar_name in "mnli_dataset_in_distribution"
    do
        for run_seed in "0"
        do

            for model_name in "EleutherAI/pythia-1b" \
                            # "Qwen/Qwen2.5-7B" \
                              
                    

            do
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
                training_pcfg_nlp.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 50 \
                --data_seed 5 \
                --run_seed ${run_seed} \
                --comment ${comment} \
                --considered_training_samples ${considered_training_samples} \
                --considered_eval_samples 128 \
                --store_result \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --nlp_dataset \
                --add_instruction

                # --use_deepspeed \


            done
        done
    done
done
