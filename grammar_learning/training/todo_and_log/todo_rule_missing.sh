comment="Week_2024.08.26_rule_missing"

# gpt2-large
for grammar_name in "pcfg_cfg3b_disjoint_terminals_one_rule_missing"
do
    # pretrain model
    time torchrun --nproc-per-node 2 training.py \
    --model_name gpt2-large \
    --grammar_name ${grammar_name} \
    --num_samples 10000 \
    --considered_training_samples 1000 \
    --num_train_epochs 50 \
    --given_seed 5 \
    --batch_size 8 \
    --comment ${comment} \
    --store_result \
    --learning_rate 0.00005 \
    # --include_edit_distance_eval \
    # --include_incorrect_random_eval \
    # --combine_edit_distance \
done

