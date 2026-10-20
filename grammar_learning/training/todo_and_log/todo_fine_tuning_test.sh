#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=200GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-3a100grid-01


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=2
export HF_HUB_CACHE=/path/to/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/path/to/shared/huggingface_cache/datasets

workdir="/path/to/training"
cd $workdir




nvidia-smi



comment="fine_tuning"




for considered_training_samples in "16"
do


    # for run_seed in "0" "1" "2"
    # for run_seed in "1" "2"
    for run_seed in "0"
    do


        for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
        do


            for model_name in \
                    "EleutherAI/pythia-1b" \
                    # "meta-llama/Llama-3.2-1B"
                    # Qwen/Qwen2.5-1.5B \
                    # "mistralai/Mistral-7B-v0.3" \
                    # "google/gemma-2-2b" \
                    

            
            do

                for memorization_algo in \
                                        "no_intervention"  \
                                        # "training_test_equal_variance"
                                        # "deduplication" \
                                        # "remove_after_memorized" \
                                        
                                         
                       

                do


                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 5 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_eval_samples 128 \
                    --store_result \
                    --memorization_algo ${memorization_algo} \



                done
            done
        done
    done
done






