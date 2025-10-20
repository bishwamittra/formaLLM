nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_07_11_nlp_dataset_fine_tuning"






for run_seed in "0"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "mistralai/Mistral-7B-v0.3" \
                        "Qwen/Qwen2.5-7B" \
                        
                        
                        

        do

        



            for grammar_name in "mnli_dataset_out_distribution"
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
                --nlp_dataset \
                --add_instruction

                


                
                

            done
        done
    done
done











for run_seed in "0"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        "base_models_vnanda/Llama-2-13b-hf" \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "base_models_soumi/opt-model-6.7B" \

                        
                        

        do

        



            for grammar_name in "mnli_dataset_out_distribution"
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
                --nlp_dataset \
                --add_instruction

                


                
                

            done
        done
    done
done











