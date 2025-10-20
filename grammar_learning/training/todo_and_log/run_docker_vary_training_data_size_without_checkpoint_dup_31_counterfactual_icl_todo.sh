nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_09_16_cf_with_memorization_mitigation_fine_tuning"



for run_seed in "0"
do


    for memorization_algo in "no_intervention" \
                             "deduplication" \
        
    do        

        for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob_counterfactual"
        do




            # for considered_training_samples in "16" "64" "1024"
            for considered_training_samples in "16" "64"
            do

                for model_name in "mistralai/Mistral-7B-v0.3" \
                        # "EleutherAI/pythia-1b" \
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
                    --memorization_algo ${memorization_algo} \
                    --use_deepspeed
                    

                    
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
                        --counterfactual_memorization \
                        --counterfactual_string_index ${counterfactual_string_index} \
                        --memorization_algo ${memorization_algo} \
                        --use_deepspeed
                        

                    done                    
                done
            done
        done
    done
done









