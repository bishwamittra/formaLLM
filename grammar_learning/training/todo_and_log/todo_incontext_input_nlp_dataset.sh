#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-05


#SBATCH -o %x_%j.out      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="../training"
cd $workdir
nvidia-smi




comment="2025_07_11_nlp_dataset_incontext_input"



# for grammar_name in "rte_dataset_in_distribution"

for grammar_name in "mnli_dataset_in_distribution"
do
    for run_seed in "0" "1" "2"
    # for run_seed in "0"
    do
        
        for considered_incontext_examples in "16" "64" "256" "1024"
        # for considered_incontext_examples in "16"
        do
        
            for model_name in \
                            "opt-model-6.7B" \
                            "Llama-2-7b-hf" \
                            
                            
                            # "mistralai/Mistral-Nemo-Base-2407"\
                            # "Llama-2-13b-hf" \
                            # "meta-llama/Meta-Llama-3.1-8B" \
                            
                            
                            # "mistralai/Mistral-7B-v0.3" \
                            # "Qwen/Qwen2.5-7B" \
                            
                            # "google/gemma-2-9b" \
                            # "meta-llama/Meta-Llama-3.1-8B" \
                            # "Llama-2-7b-hf" \
                            # "opt-model-6.7B" \
                            # "EleutherAI/pythia-6.9b" \
                            


                            # "Qwen/Qwen2.5-1.5B" \
                            # "EleutherAI/pythia-1b" \
                            # "Qwen/Qwen2.5-14B" \
                            # "google/gemma-2-2b" \
                            # "meta-llama/Llama-3.2-1B" \
                            # "meta-llama/Llama-3.2-3B" \
                            # "opt-model-1.3B" \
                            # "opt-model-2.7B" \
                            # "opt-model-6.7B" \
                            # "EleutherAI/pythia-2.8b" \

                            
            do
                time TRANSFORMERS_VERBOSITY=error python training_pcfg_nlp.py \
                --inference_only_mode \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples ${considered_incontext_examples} \
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



for grammar_name in "mnli_dataset_out_distribution"
do
    for run_seed in "0" "1" "2"
    # for run_seed in "0"
    do
        
        for considered_incontext_examples in "16" "64" "256" "1024"
        # for considered_incontext_examples in "16"
        do
        
            for model_name in \
                            "opt-model-6.7B" \
                            "mistralai/Mistral-Nemo-Base-2407"\
                            "Llama-2-13b-hf" \
                            "meta-llama/Meta-Llama-3.1-8B" \
                            "mistralai/Mistral-7B-v0.3" \
                            "Qwen/Qwen2.5-7B" \
                            "Llama-2-7b-hf" \
                            
                            
                            # "google/gemma-2-9b" \
                            # "meta-llama/Meta-Llama-3.1-8B" \
                            # "Llama-2-7b-hf" \
                            # "opt-model-6.7B" \
                            # "EleutherAI/pythia-6.9b" \
                            


                            # "Qwen/Qwen2.5-1.5B" \
                            # "EleutherAI/pythia-1b" \
                            # "Qwen/Qwen2.5-14B" \
                            # "google/gemma-2-2b" \
                            # "meta-llama/Llama-3.2-1B" \
                            # "meta-llama/Llama-3.2-3B" \
                            # "opt-model-1.3B" \
                            # "opt-model-2.7B" \
                            # "opt-model-6.7B" \
                            # "Llama-2-7b-hf" \
                            # "EleutherAI/pythia-2.8b" \

                            
            do
                time TRANSFORMERS_VERBOSITY=error python training_pcfg_nlp.py \
                --inference_only_mode \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples ${considered_incontext_examples} \
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




