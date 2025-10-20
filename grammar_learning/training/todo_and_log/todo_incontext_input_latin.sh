#!/bin/bash
#
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-3a100grid-01,sws-8a100-03,sws-8a100-01,sws-8h100grid-07

# sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03,sws-8h100grid-07


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/hub/
export HF_DATASETS_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/datasets/


workdir="/NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi




comment="2024_12_09_g4"

# for grammar_name in "pcfg_cfg3b_disjoint_terminals_latin"
for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
do
    for considered_incontext_examples in "0" "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
    do
        for run_seed in "0" "1" "2"
        do
            for model_name in "google/gemma-2-2b" \
                              "google/gemma-2-9b" \
                              "mistralai/Mistral-Nemo-Base-2407"\
                              "mistralai/Mistral-7B-v0.3" \
                              "EleutherAI/pythia-6.9b" \
                              "EleutherAI/pythia-1b" \
                              "EleutherAI/pythia-2.8b" \
                              "meta-llama/Meta-Llama-3.1-8B" \
                              "meta-llama/Meta-Llama-3-8B" \
                              "meta-llama/Llama-3.2-1B" \
                              "meta-llama/Llama-3.2-3B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-1.3B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-2.7B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-6.7B" \
                              "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf" \
                              "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf" \

                              
            do
                time torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
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
                # --include_edit_distance_eval \
                # --include_incorrect_random_eval \
                # --combine_edit_distance

            done
        done
    done
done



