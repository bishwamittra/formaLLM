nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="comment="2025_06_13_counterfactural_memorization_fine_tuning""


for run_seed in "0" "1" "2"
# for run_seed in "0"
# for run_seed in "1"
do

    # for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob_counterfactual" "pcfg_cfg3b_eq_len_skewed_prob_0.90_counterfactual"
    # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob_counterfactual"
    # for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob_counterfactual"
    # for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_all_rules_skewed_prob_counterfactual"
    # for grammar_name in "preg_alphabet_combined_counterfactual"
    for grammar_name in "preg_alphabet_combined_varying_freq_counterfactual"
    do




        # for considered_training_samples in "16" "64" "256" "1024"
        # for considered_training_samples in "16" "64" "256"
        for considered_training_samples in "256"
        do

            for model_name in "mistralai/Mistral-7B-v0.3" \
                    # "google/gemma-2-9b" \
                    # "EleutherAI/pythia-6.9b" \
                    # "EleutherAI/pythia-1b" \

            do

                # excluding the string
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
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --use_deepspeed \


                
                # including the string
                for counterfactual_string_index in "0" "1" "2"
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
                    --include_incorrect_random_eval \
                    --include_edit_distance_eval \
                    --combine_edit_distance \
                    --counterfactual_memorization \
                    --counterfactual_string_index ${counterfactual_string_index} \
                    --use_deepspeed \
                    
                    

                done



                
                
                

            done
        done
    done
done




