comment="Week_2024.08.09_generation_prob_based_separation"

for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
do
    for fraction_train in "0.125"
    do
        # pretrain model
        time torchrun --nproc-per-node 2 training.py \
        --inference_only_mode \
        --model_name pythia-1b \
        --checkpoint_path_overwrite /NS/llm-1/nobackup/vnanda/llm_base_models/pythia-1b \
        --grammar_name ${grammar_name} \
        --data_comment "gen_prob" \
        --num_train_samples 8000 \
        --num_test_samples 42000 \
        --considered_training_samples 1000 \
        --num_train_epochs 1 \
        --given_seed 5 \
        --batch_size 8 \
        --comment ${comment} \
        --store_result \
        --include_edit_distance_eval \
        --include_incorrect_random_eval \
        # --combine_edit_distance \
    done
done

