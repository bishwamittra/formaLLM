nvidia-smi

export GPUS_PER_NODE=4
cd training


comment="2025_04_04_rebuttal_fine_tuning"


for run_seed in "0" "1" "2"
do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals" "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_cfg3b_disjoint_terminals_latin" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
    do


    
        

        for model_name in "Qwen/Qwen2.5-0.5B" \
                          "Qwen/Qwen2.5-1.5B" \
                          "Qwen/Qwen2.5-7B" \
                          "Qwen/Qwen2.5-14B" \
                        


        do                                                      


            
            for considered_training_samples in "1" "16" "64" "256" "1024"
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








        # for model_name in "mistralai/Mistral-7B-v0.3" \
        #                 "google/gemma-2-9b" \
        #                 "base_models_vnanda/Llama-2-7b-hf" \
        #                 "EleutherAI/pythia-6.9b" \
        #                 "meta-llama/Meta-Llama-3.1-8B" \
        #                 "base_models_soumi/opt-model-6.7B" \
        #                 "google/gemma-2-2b" \
        #                 "EleutherAI/pythia-1b" \
        #                 "EleutherAI/pythia-2.8b" \
        #                 "meta-llama/Llama-3.2-1B" \
        #                 "meta-llama/Llama-3.2-3B" \
        #                 "base_models_soumi/opt-model-1.3B" \
        #                 "base_models_soumi/opt-model-2.7B" \
        #                 "base_models_vnanda/Llama-2-13b-hf" \
        #                 "mistralai/Mistral-Nemo-Base-2407"\
