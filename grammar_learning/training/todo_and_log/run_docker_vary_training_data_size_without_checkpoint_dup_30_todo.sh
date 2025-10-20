nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_09_16_memorization_based_intervention_fine_tuning"




for considered_training_samples in "256"
# for considered_training_samples in "16" "64"
# for considered_training_samples in "256" "1024"
do


    for run_seed in "0" "1" "2"
    # for run_seed in "0"
    do


        # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob_0.75" 
        for grammar_name in ""pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob""
        # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg_extended_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_uniform_prob" "pcfg_cfg3b_eq_len_uniform_prob"
        # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob" "pcfg_cfg3b_eq_len_skewed_prob"
        # for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg_extended_eq_len_skewed_prob" "pcfg_cfg3b_disjoint_terminals_skewed_prob"
        do


            for model_name in \
                    "mistralai/Mistral-7B-v0.3" \
                    "EleutherAI/pythia-1b" \
                    # "meta-llama/Llama-3.2-1B"
                    # Qwen/Qwen2.5-1.5B \
                    # "google/gemma-2-2b" \
                    

            
            do

                for memorization_algo in \
                                        "no_intervention"  \
                                        "deduplication" \
                                        "training_test_equal_variance" \
                                        "remove_after_memorized" \
                                        # "remove_after_memorized_edit_distance" \
                                        # "tail_distribution" \
                                        # "training_variance" \
                                        # "remove_after_memorized_and_never_put_back" \
                                        
                                         
                       

                do


                    time torchrun --nproc_per_node=$GPUS_PER_NODE \
                    training.py \
                    --model_name ${model_name} \
                    --grammar_name ${grammar_name} \
                    --num_samples 10000 \
                    --num_train_epochs 50 \
                    --data_seed 5 \
                    --run_seed ${run_seed} \
                    --comment ${comment} \
                    --considered_training_samples ${considered_training_samples} \
                    --considered_eval_samples 1024 \
                    --store_result \
                    --memorization_algo ${memorization_algo} \
                    --use_deepspeed \
                    # --include_incorrect_random_eval \
                    # --include_edit_distance_eval \
                    # --combine_edit_distance \



                done
            done
        done
    done
done



