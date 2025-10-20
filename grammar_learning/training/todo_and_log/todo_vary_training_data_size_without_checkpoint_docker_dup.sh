nvidia-smi

export GPUS_PER_NODE=4
cd training


comment="2025_02_03_entropy"




for run_seed in "0" "1" "2"
do

    for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
    do

        for grammar_name in "pcfg_cfg3b_disjoint_terminals_all_rules_0.70" \
                            "pcfg_cfg3b_disjoint_terminals_all_rules_0.90" \


        do


            for model_name in   "mistralai/Mistral-7B-v0.3" \
                                "google/gemma-2-9b" \
                                "EleutherAI/pythia-6.9b" \
                            


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





