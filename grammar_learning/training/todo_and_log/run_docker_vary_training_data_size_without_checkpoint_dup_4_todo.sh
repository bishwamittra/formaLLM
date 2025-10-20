nvidia-smi

export GPUS_PER_NODE=4
cd training


comment="2025_02_25_probabilistic_context_sensitive_grammars_fine_tuning"



for run_seed in "0" "1" "2"
do


    # for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
    for considered_training_samples in "1" "4" "16" "64" "256" "1024"
    do


        

        for model_name in "mistralai/Mistral-7B-v0.3" \
                        "google/gemma-2-9b" \
                        "EleutherAI/pythia-6.9b" \
                        "base_models_soumi/opt-model-6.7B" \
                        


        do                                                      


            for grammar_name in "pcsg_csg3b_disjoint_terminals_A8_left" "pcsg_csg3b_disjoint_terminals_A8_right"
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

