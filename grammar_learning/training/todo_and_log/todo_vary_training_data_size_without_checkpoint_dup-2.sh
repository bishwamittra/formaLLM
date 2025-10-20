#!/bin/bash
#
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres=gpu:2           # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 7-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-2a40grid-01



#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written

export GPUS_PER_NODE=2
export HF_HUB_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/hub/
export HF_DATASETS_CACHE=/NS/llm-1/nobackup/shared/huggingface_cache/datasets/

workdir="/NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training"
cd $workdir



nvidia-smi

comment="2024_11_27_learning_scheduler"









for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"


do
    for considered_training_samples in "8"
    do
        for run_seed in "0" "1" "2"
        do
            for model_name in "EleutherAI/pythia-1b" \
                              "meta-llama/Llama-3.2-1B" \


            do

                for lr_schedular in "linear" "cosine" "constant"
                do
                    
                    time torchrun \
                    --nproc_per_node=$GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 100 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_eval_samples 100 \
                    --store_result \
                    --use_deepspeed \
                    --lr_scheduler ${lr_schedular} \
                    --warmup_ratio 0 \
                
    
                done        
            done
        done 
    done
done
