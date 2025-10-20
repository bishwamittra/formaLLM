nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="comment="2025_03_27_benchmark_probabilistic_finite_state_automaton_fine_tuning""


for run_seed in "0" "1" "2"
do

    for considered_training_samples in "64" "1024"
    do

        for grammar_name in "pfsa_states_8_symbols_5_index_2_alphabet_0-1-2-3" \
                        "pfsa_states_4_symbols_9_index_3_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_5_symbols_5_index_4_alphabet_0-1-2-3" \
                        "pfsa_states_16_symbols_5_index_4_alphabet_0-1-2-3" \
                        "pfsa_states_12_symbols_7_index_2_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_11_index_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_4_symbols_9_index_0_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_5_index_3_alphabet_0-1-2-3" \
                        "pfsa_states_8_symbols_7_index_6_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_5_index_0_alphabet_0-1-2-3" \
                        "pfsa_states_12_symbols_11_index_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_4_symbols_9_index_1_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_4_symbols_11_index_3_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_11_index_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_5_index_3_alphabet_0-1-2-3" \
                        "pfsa_states_8_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_5_index_0_alphabet_0-1-2-3" \
                        "pfsa_states_4_symbols_11_index_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_11_index_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_2_symbols_3_index_0_alphabet_0-1" \
                        "pfsa_states_16_symbols_7_index_2_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_4_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_5_index_1_alphabet_0-1-2-3" \
                        "pfsa_states_12_symbols_11_index_3_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_4_symbols_11_index_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_5_index_1_alphabet_0-1-2-3" \
                        "pfsa_states_5_symbols_5_index_1_alphabet_0-1-2-3" \
                        "pfsa_states_16_symbols_7_index_5_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_5_index_4_alphabet_0-1-2-3" \
                        "pfsa_states_2_symbols_11_index_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_11_index_9_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_11_index_3_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_5_index_0_alphabet_0-1-2-3" \
                        "pfsa_states_5_symbols_5_index_0_alphabet_0-1-2-3" \
                        "pfsa_states_5_symbols_5_index_3_alphabet_0-1-2-3" \
                        "pfsa_states_16_symbols_5_index_3_alphabet_0-1-2-3" \
                        "pfsa_states_8_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_4_symbols_11_index_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_11_index_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_5_index_2_alphabet_0-1-2-3" \
                        "pfsa_states_4_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_2_symbols_9_index_0_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_5_index_2_alphabet_0-1-2-3" \
                        "pfsa_states_8_symbols_11_index_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_8_symbols_5_index_4_alphabet_0-1-2-3" \
                        "pfsa_states_16_symbols_11_index_0_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_11_index_4_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_3_index_0_alphabet_0-1" \
                        "pfsa_states_5_symbols_7_index_2_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_7_index_4_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_9_index_8_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_2_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_11_index_4_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_7_index_6_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_9_index_3_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_7_index_0_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_4_symbols_7_index_3_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_11_index_10_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_11_index_5_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_11_index_10_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_11_index_1_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_8_symbols_7_index_2_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_9_index_0_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_12_symbols_9_index_1_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_11_index_5_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_5_symbols_7_index_4_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_9_index_4_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_7_index_5_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_12_symbols_3_index_1_alphabet_0-1" \
                        "pfsa_states_8_symbols_9_index_2_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_9_index_3_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_4_symbols_5_index_2_alphabet_0-1-2-3" \
                        "pfsa_states_12_symbols_11_index_8_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_9_index_5_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_11_index_6_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_3_index_0_alphabet_0-1" \
                        "pfsa_states_8_symbols_9_index_1_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_12_symbols_3_index_2_alphabet_0-1" \
                        "pfsa_states_16_symbols_9_index_7_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_9_index_6_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_12_symbols_11_index_6_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_12_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_11_index_2_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_8_symbols_9_index_0_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_3_index_0_alphabet_0-1" \
                        "pfsa_states_4_symbols_5_index_3_alphabet_0-1-2-3" \
                        "pfsa_states_2_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_8_symbols_9_index_5_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_9_index_3_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_4_symbols_3_index_0_alphabet_0-1" \
                        "pfsa_states_12_symbols_11_index_9_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_16_symbols_9_index_2_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_9_index_4_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_4_symbols_7_index_2_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_7_index_1_alphabet_0-1-2-3-4-5" \
                        "pfsa_states_16_symbols_9_index_0_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_8_symbols_9_index_6_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_12_symbols_11_index_7_alphabet_0-1-2-3-4-5-6-7-8-9" \
                        "pfsa_states_8_symbols_9_index_7_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_16_symbols_9_index_1_alphabet_0-1-2-3-4-5-6-7" \
                        "pfsa_states_4_symbols_3_index_2_alphabet_0-1" \
                        "pfsa_states_8_symbols_7_index_3_alphabet_0-1-2-3-4-5" \

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







