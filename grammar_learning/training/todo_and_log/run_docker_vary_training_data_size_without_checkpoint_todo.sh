nvidia-smi

export GPUS_PER_NODE=4
cd training






comment="2024_02_17_fine_tuning_g2_grammar_recognizer_missing"





for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
do
        for run_seed in "1"
        do
        for model_name in \
                        "base_models_soumi/opt-model-6.7B" \
                        "base_models_vnanda/Llama-2-13b-hf" \
                        


        do                                                      
            for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
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


for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
do
        for run_seed in "2"
        do
        for model_name in \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        
                        


        do                                                      
            for considered_training_samples in "4" "8" "16" "32" "64" "128" "256" "512" "1024"
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














comment="2024_02_17_fine_tuning_g1_grammar_recognizer_missing"


for model_name in \
                "meta-llama/Llama-3.2-1B" \
                "meta-llama/Llama-3.2-3B" \
                "base_models_soumi/opt-model-1.3B" \
                "base_models_soumi/opt-model-2.7B" \
                "EleutherAI/pythia-1b" \
                "EleutherAI/pythia-2.8b" \
                "base_models_soumi/opt-model-6.7B" \
                "base_models_vnanda/Llama-2-13b-hf" \
                "meta-llama/Meta-Llama-3.1-8B" \
                


do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals"
    do


            for run_seed in  "2"
            do
                        


                                                              
            for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
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


