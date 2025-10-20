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


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/hub/
export HF_DATASETS_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/datasets/
workdir="/NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training"
cd $workdir



comment="Week_2024.11.20_double_descent"

# for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
for considered_training_samples in "1024"
do
    for grammar_name in "pcfg_cfg3b_disjoint_terminals" "pcfg_cfg3b_disjoint_terminals_latin" "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_cfg3b" "pcfg_balanced_parenthesis"
    do
        for run_seed in "0"
        do

            for model_name in "google/gemma-2-2b" \
                              "EleutherAI/pythia-1b" \
                              "meta-llama/Llama-3.2-1B" \
                              "/NS/llm-1/nobackup/soumi/opt-model-1.3B" \
                              
                    

            do
                # pretrain model
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                training.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 10000 \
                --data_seed 5 \
                --considered_eval_samples 1024 \
                --batch_size 8 \
                --run_seed ${run_seed} \
                --comment ${comment} \
                --considered_training_samples ${considered_training_samples} \
                --store_result \


            done
        done
    done
done
