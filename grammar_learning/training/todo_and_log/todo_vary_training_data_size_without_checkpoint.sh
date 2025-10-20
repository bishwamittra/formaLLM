#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4           # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
# SBATCH --exclude=sws-8a100-02


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/hub/
export HF_DATASETS_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/datasets/

workdir="/NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training"
cd $workdir


nvidia-smi


comment="2024_12_09_fine_tuning_g3_g4"




# requires docker

#   "mistralai/Mistral-Nemo-Base-2407"\
#   "mistralai/Mistral-7B-v0.3" \
#   "meta-llama/Meta-Llama-3.1-8B" \
#   "meta-llama/Meta-Llama-3-8B" \
#   "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf" \




for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin" "pcfg_cfg3b_disjoint_terminals_latin"
do
    for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
    do
        for run_seed in "0" "1" "2"
        do
            for model_name in "google/gemma-2-2b" \
                              "google/gemma-2-9b" \
                              "EleutherAI/pythia-6.9b" \
                              "EleutherAI/pythia-1b" \
                              "EleutherAI/pythia-2.8b" \
                              "meta-llama/Llama-3.2-1B" \
                              "meta-llama/Llama-3.2-3B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-1.3B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-2.7B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-6.7B" \
                              "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf" \
                              
                    

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
                --considered_eval_samples 128 \
                --store_result \
                --use_deepspeed \
                
  
            
            done
        done 
    done
done



# comment="2024_11_25_hierarchical_cfg_training_and_test_only"









# for grammar_name in "pcfg_2_3_2_10_latin" \
#                 "pcfg_2_3_2_10_numerical" \
#                 "pcfg_2_5_2_10_latin" \
#                 "pcfg_2_5_2_10_numerical" \
#                 "pcfg_3_3_2_10_latin" \
#                 "pcfg_3_3_2_10_numerical" \
#                 "pcfg_3_5_2_10_latin" \
#                 "pcfg_3_5_2_10_numerical" \
#                 "pcfg_5_3_2_10_latin" \
#                 "pcfg_5_3_2_10_numerical" \
#                 "pcfg_5_5_2_10_latin" \
#                 "pcfg_5_5_2_10_numerical" \
#                 "pcfg_10_3_2_10_latin" \
#                 "pcfg_10_3_2_10_numerical" \
#                 "pcfg_10_5_2_10_latin" \


# do
#     for considered_training_samples in "1024"
#     do
#         for run_seed in "0" "1" "2"
#         do
#             for model_name in  "EleutherAI/pythia-1b" \
#                             # "google/gemma-2-2b" \
#                             #   "meta-llama/Llama-3.2-1B" \
#                             #   "/NS/llm-1/nobackup/soumi/opt-model-1.3B" \
                              
                    

#             do
#                 time torchrun \
#                 --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
#                 training.py \
#                 --model_name ${model_name} \
#                 --grammar_name ${grammar_name} \
#                 --num_samples 10000 \
#                 --num_train_epochs 50 \
#                 --data_seed 5 \
#                 --run_seed ${run_seed} \
#                 --comment ${comment} \
#                 --considered_training_samples ${considered_training_samples} \
#                 --considered_eval_samples 1024 \
#                 --store_result \
#                 --use_deepspeed \
                
  
            
#             done
#         done 
#     done
# done
