#!/bin/bash
#
#SBATCH --partition=a100       # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8a100-02,sws-8a100-03,sws-8a100-01


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi




comment="2025_03_10_benchmark_hierarchical_probabilistic_context_free_languages_incontext_input_duplicate"


for considered_incontext_examples in "1" "4"
do


    for run_seed in "0" "1" "2"
    do



        for grammar_name in "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_2_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_16_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_4_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_16_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_3_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_2_rules_4_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_3_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_2_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_4_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_4_skewness_1_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_4_rules_2_skewness_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_16_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_2_rules_3_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_2_max-breadth_8_rules_4_skewness_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                            "pcfg_max-depth_4_max-breadth_8_rules_2_skewness_2_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_8_rules_3_skewness_0_alphabet_0-1-2-3-4" \
                            "pcfg_max-depth_4_max-breadth_2_rules_2_skewness_1_alphabet_0-1-2-3-4" \




        do


            for model_name in "google/gemma-2-9b" \
                              "mistralai/Mistral-7B-v0.3" \
                              "EleutherAI/pythia-6.9b" \
                              "/NS/llm-1/nobackup/soumi/opt-model-6.7B" \
                              
                              
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
                --include_edit_distance_eval \
                --include_incorrect_random_eval \
                --combine_edit_distance

                

            done
        done
    done
done



