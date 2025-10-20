nvidia-smi

export GPUS_PER_NODE=4
cd training






comment="2025_09_06_memorization_mitigation_test_ood_fine_tuning"











for run_seed in "0" "1" "2"
do

    for considered_training_samples in "256"
    # for considered_training_samples in "16" "64" "256" "1024"
    # for considered_training_samples in "16" "64" "256" "1024"
    do


        for training_grammar_name in "pcfg_cfg3b_eq_len_uniform_prob"
        # for training_grammar_name in "pcfg_cfg3b_disjoint_terminals"
        do


            for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob"
            # for grammar_name in "pcfg_cfg3b_disjoint_terminals_skewed_prob"
            do
            

                for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        "EleutherAI/pythia-1b" \
                        # "google/gemma-2-2b" \

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
                    --incontext_data_source ${training_grammar_name}:train_sequences \
                    --considered_eval_samples 128 \
                    --store_result \
                    --use_deepspeed \
                    # --include_incorrect_random_eval \
                    # --include_edit_distance_eval \
                    # --combine_edit_distance \
                    # --include_grammar_edit_eval \
                    

                done
            done
        done
    done
done








