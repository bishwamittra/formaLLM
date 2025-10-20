
# duplicating training samples
comment="Week_2024.08.09_training_strategy"

for grammar_name in "pcfg_cfg3b_disjoint_terminals"
do
    for fraction_train in "0.128"
    do
        for config in "32,32" "64,16" "128,8" "256,4" "512,2" "1024,1"
        # for config in "512,2" "1024,1"
        do
            IFS=","
            set -- $config

            # # pretrain model
            time torchrun --nproc-per-node 2 training.py \
            --model_name gpt2-large \
            --grammar_name ${grammar_name} \
            --data_comment "training_strategy_${1}_${2}" \
            --num_train_samples 8000 \
            --num_test_samples 2000 \
            --num_train_epochs 10 \
            --given_seed 5 \
            --batch_size 8 \
            --comment ${comment} \
            --fraction_train ${fraction_train} \
            --store_result \
            --learning_rate 0.00005 \
            # --include_edit_distance_eval \
            # --include_incorrect_random_eval \
            # --combine_edit_distance \
        done
    done
done


# Without duplication
comment="Week_2024.08.09_training_strategy_without_duplication"
for grammar_name in "pcfg_cfg3b_disjoint_terminals"
do
    # for fraction_train in "0.128" "0.064" "0.032" "0.016" "0.008" "0.004"
    # for fraction_train in "0.128" "0.064"
    # for config in "0.128,10" "0.064,20"
    for config in "0.128,10" "0.064,20" "0.032,40" "0.016,80" "0.008,160" "0.004,320"
    do
        
        IFS=","
        set -- $config

        # # pretrain model
        time torchrun --nproc-per-node 2 training.py \
        --model_name gpt2-large \
        --grammar_name ${grammar_name} \
        --data_comment "training_strategy_1024_1" \
        --num_train_samples 8000 \
        --num_test_samples 2000 \
        --num_train_epochs ${2} \
        --given_seed 5 \
        --batch_size 8 \
        --comment ${comment} \
        --fraction_train ${1} \
        --store_result \
        --learning_rate 0.00005 \
        # --include_edit_distance_eval \
        # --include_incorrect_random_eval \
        # --combine_edit_distance \
        
    done
done