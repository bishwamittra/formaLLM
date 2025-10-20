nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_08_14_sensitive_vs_nonsensitive_tokens_fine_tuning"

for considered_training_samples in "512"
do
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity"
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_3"
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_3_deduplicated"
    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_8" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_9"

    # for grammar_name in "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_0_deduplicated" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_1_deduplicated" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_2_deduplicated" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_3_deduplicated" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_4_deduplicated" \
    #                     "pcfg_cfg3b_disjoint_terminals_sensitivity_modification_10_5_deduplicated" \


    for grammar_name in "pii_to_pii_high_to_low_entropy" \
                        "pii_to_pii_low_to_high_entropy"

    do
        for run_seed in "0"
        do

            for model_name in "mistralai/Mistral-7B-v0.3" \
                            
                                                         
                    

            do
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
                training.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 100 \
                --data_seed 5 \
                --run_seed ${run_seed} \
                --comment ${comment} \
                --considered_training_samples ${considered_training_samples} \
                --considered_eval_samples 512 \
                --store_result \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --use_deepspeed \


            done
        done
    done
done
