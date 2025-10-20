nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_05_11_fixed_parameters_counterfactural_memorization_incontext_input_multiple_models"



for run_seed in "0" "1" "2"
do

    for considered_incontext_examples in "0"
    do

        
        # for grammar_name in  "pcfg_cfg3b_eq_len_skewed_prob_counterfactual" "pcfg_cfg3b_eq_len_skewed_prob_0.90_counterfactual" "pcfg_cfg3b_eq_len_uniform_prob_counterfactual"
        for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_all_rules_skewed_prob_counterfactual" "pcfg_cfg3b_eq_len_skewed_prob_counterfactual" "preg_alphabet_combined_counterfactual"

        do




            for considered_training_samples in "16" "64" "256" "1024"
            do

                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-1b" \
                        # "google/gemma-2-9b" \
                        # "EleutherAI/pythia-6.9b" \
                        # "EleutherAI/pythia-1b" \

                do

                    # excluding the string
                    time TRANSFORMERS_VERBOSITY=error python training.py \
                    --inference_only_mode \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 1 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --incontext_input \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_incontext_examples ${considered_incontext_examples} \
                    --considered_eval_samples 128 \
                    --store_result \
                    --batch_size 1 \
                    --include_incorrect_random_eval \
                    --include_edit_distance_eval \
                    --combine_edit_distance \
                    


                    
                    # including the string
                    for counterfactual_string_index in "0" "1" "2"
                    do

                        
                        time TRANSFORMERS_VERBOSITY=error python training.py \
                        --inference_only_mode \
                        --model_name ${model_name} \
                        --grammar_name ${grammar_name} \
                        --num_samples 10000 \
                        --num_train_epochs 1 \
                        --data_seed 5 \
                        --run_seed ${run_seed} \
                        --comment ${comment} \
                        --incontext_input \
                        --considered_training_samples ${considered_training_samples} \
                        --considered_incontext_examples ${considered_incontext_examples} \
                        --considered_eval_samples 128 \
                        --store_result \
                        --batch_size 1 \
                        --include_incorrect_random_eval \
                        --include_edit_distance_eval \
                        --combine_edit_distance \
                        --counterfactual_memorization \
                        --counterfactual_string_index ${counterfactual_string_index} \
                        
                        
                        

                    done



                    
                    
                    

                done
            done
        done
    done
done






