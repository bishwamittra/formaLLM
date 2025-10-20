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



comment="2025_05_27_memorization_with_local_prefix_exp_fine_tuning"


for run_seed in "0"
do

    for considered_training_samples in "64"
    # for considered_training_samples in "16" "256"
    do


        for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" 
        # for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_all_rules_skewed_prob"
        do

            for model_name in \
                    "EleutherAI/pythia-1b" \
                    # "google/gemma-2-2b" \
                    # "mistralai/Mistral-7B-v0.3" \

            do


                # for global_prefix_config in 'random_token' 'same_language' 'no_global_prefix'
                for global_prefix_config in 'no_global_prefix'
                do

                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 100 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_eval_samples 1024 \
                    --store_result \
                    --generate_text \
                    --max_new_tokens 1 \
                    --compute_msp \
                    --global_prefix_config ${global_prefix_config}
        


            
                done                
            done
        done
    done
done





# checkpoint_folder="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training/artifacts/output_2025_05_27_08_03_mistralai_Mistral-7B-v0.3_pcfg_cfg3b_eq_len_skewed_prob_10000_0_2025_05_05_save_checkpoint_fine_tuning_16"

                
# for checkpoint_path_overwrite in ${checkpoint_folder}/checkpoint*
# do

#     echo "checkpoint_path_overwrite: ${checkpoint_path_overwrite}"

#     time TRANSFORMERS_VERBOSITY=error python \
#     training.py \
#     --model_name ${model_name} \
#     --grammar_name ${grammar_name} \
#     --num_samples 10000 \
#     --num_train_epochs 50 \
#     --data_seed 5 \
#     --run_seed ${run_seed} \
#     --comment ${comment} \
#     --considered_training_samples ${considered_training_samples} \
#     --considered_eval_samples ${considered_training_samples} \
#     --store_result \
#     --generate_text \
#     --max_new_tokens 1 \
#     --compute_msp \
#     --checkpoint_path_overwrite ${checkpoint_path_overwrite}


# done