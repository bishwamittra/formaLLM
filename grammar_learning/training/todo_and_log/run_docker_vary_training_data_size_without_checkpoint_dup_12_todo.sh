nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="comment="2025_03_27_benchmark_hierarchical_probabilistic_context_free_languages_fine_tuning""


for run_seed in "0" "1" "2"
do

    for considered_training_samples in "64" "1024"
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


        

            for model_name in \
                    "google/gemma-2-9b" \
                    "EleutherAI/pythia-6.9b" \
                    "base_models_soumi/opt-model-6.7B" \
                    "base_models_vnanda/Llama-2-7b-hf" \
                    "meta-llama/Meta-Llama-3.1-8B" \

                    

            do

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
                --considered_eval_samples 128 \
                --store_result \
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                

            done
        done
    done
done







