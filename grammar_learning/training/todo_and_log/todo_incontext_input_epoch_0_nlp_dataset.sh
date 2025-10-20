#!/bin/bash
#
#SBATCH --partition=h100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h200grid-01,sws-8h200grid-02


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi




comment="2025_07_11_nlp_dataset_incontext_input"



for grammar_name in "mnli_dataset_in_distribution" "mnli_dataset_out_distribution"
do
    for considered_incontext_examples in "0"
    do
        
        for considered_training_samples in "1" "4" "16" "64" "256" "1024"
        do
            for run_seed in "0" "1" "2"
            do
                for model_name in \
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-6.7B" \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-13b-hf" \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "mistralai/Mistral-7B-v0.3" \
                        "Qwen/Qwen2.5-7B" \
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-7b-hf" \
                        
                                
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
                    --considered_eval_samples 128 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --batch_size 1 \
                    --comment ${comment} \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --incontext_separator "double_newline" \
                    --nlp_dataset \
                    --add_instruction



                done
            done
        done
    done
done



