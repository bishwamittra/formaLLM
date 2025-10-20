#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
# SBATCH --exclude=sws-8a100-03,sws-8a100-04
# SBATCH --exclude=sws-8h100grid-05,sws-8h100grid-07


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi


comment="2025_02_17_incontext_repetitions"




for grammar_name in "pcfg_cfg3b_disjoint_terminals_latin" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
do

    for considered_incontext_examples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
    do
        for run_seed in "0" "1" "2"
        do
            for model_name in "mistralai/Mistral-7B-v0.3" \
                              "google/gemma-2-9b" \
                              "EleutherAI/pythia-6.9b" \
                              
            do

                for considered_incontext_repetitions  in "2" "4" "8" "16"             
                                
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
                    --considered_incontext_repetitions ${considered_incontext_repetitions} \
                    --num_train_epochs 1 \
                    --considered_eval_samples 128 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --batch_size 1 \
                    --comment ${comment} \
                    --include_edit_distance_eval \
                    --include_incorrect_random_eval \
                    --combine_edit_distance



                done
            done
        done
    done
done



