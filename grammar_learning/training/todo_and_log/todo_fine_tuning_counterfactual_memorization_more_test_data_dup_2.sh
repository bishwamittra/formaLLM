#!/bin/bash
#
#SBATCH --partition=a40       # Use GPU partition "a100"
#SBATCH --gres=gpu:2          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-2a40grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=2
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi


"EleutherAI/pythia-2.8b" \
"meta-llama/Llama-3.2-3B" \
"/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B" \

"google/gemma-2-2b" \ ** weird

comment="2025_05_05_more_test_data_memorization_fine_tuning"


for run_seed in "0"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "Qwen/Qwen2.5-1.5B" \
                        "EleutherAI/pythia-2.8b" \
                        "google/gemma-2-2b" \
                        "meta-llama/Llama-3.2-1B" \
                        "meta-llama/Llama-3.2-3B" \
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-1.3B" \
                        "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B" \

                        # "EleutherAI/pythia-1b" \                        
                        # "Qwen/Qwen2.5-7B" \
                        # "Qwen/Qwen2.5-14B" \
                        # "EleutherAI/pythia-2.8b" \
                        # "meta-llama/Meta-Llama-3.1-8B" \
                        # "base_models_vnanda/Llama-2-13b-hf" \
                        # "base_models_vnanda/Llama-2-7b-hf" \
                        # "base_models_soumi/opt-model-6.7B" \
                        # "mistralai/Mistral-Nemo-Base-2407"\
                        

                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
            do
            

                # excluding the string
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
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
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done







