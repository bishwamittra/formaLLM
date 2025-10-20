#!/bin/bash
#
#SBATCH --partition=h100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
# SBATCH --exclude=sws-8h200grid-01,sws-8h200grid-02


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=2
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi




comment="2025_09_25_memorization_based_intervention_incontext_input"



for grammar_name in  "pcfg_cfg3b_eq_len_skewed_prob"
do
    for considered_incontext_examples in "0"
    do
        
        for considered_training_samples in "256"
        do
            for run_seed in "0" "1" "2"
            # for run_seed in "0"
            do
                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-6.9b" \
                        # "Qwen/Qwen2.5-7B" \
                    
                                                        
                do

                    for memorization_algo in "no_intervention" \
                                             "deduplication" \
            


                    do

                        time TRANSFORMERS_VERBOSITY=error python training.py \
                        --inference_only_mode \
                        --model_name ${model_name} \
                        --grammar_name ${grammar_name} \
                        --num_samples 10000 \
                        --store_result \
                        --incontext_input \
                        --considered_training_samples ${considered_training_samples} \
                        --considered_incontext_examples ${considered_incontext_examples} \
                        --num_train_epochs 1 \
                        --considered_eval_samples 1024 \
                        --data_seed 5 \
                        --run_seed ${run_seed} \
                        --batch_size 1 \
                        --comment ${comment} \
                        --memorization_algo ${memorization_algo} \
                        # --include_edit_distance_eval \
                        # --include_incorrect_random_eval \
                        # --combine_edit_distance


                    done

                done
            done
        done
    done
done



