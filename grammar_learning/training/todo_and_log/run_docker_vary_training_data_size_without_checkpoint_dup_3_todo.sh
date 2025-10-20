nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_03_04_compute_msp_text_generation_fine_tuning"


for run_seed in "0" "1" "2"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do

    
        for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob_0.75" "preg_alphabet_7" "preg_alphabet_26"
        do

            for model_name in \
                    "EleutherAI/pythia-1b" \
                    # "mistralai/Mistral-7B-v0.3" \
                    # "google/gemma-2-9b" \
                    # "EleutherAI/pythia-6.9b" \
                    

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
                --considered_eval_samples ${considered_training_samples} \
                --store_result \
                --generate_text \
                --max_new_tokens 1 \
                --compute_msp \
                

            done
        done
    done
done







# comment="2025_02_03_fine_tuning_g3_g4_grammar_recognizer"





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



# do


#     for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin" "pcfg_cfg3b_disjoint_terminals_latin"
#     do


#             for run_seed in  "2"
#             do
                        


                                                              
#             for considered_training_samples in "1" "2" "4" "8" "16" "32" "64" "128" "256" "512" "1024"
#             do
#                 time torchrun \
#                 --nproc_per_node=$GPUS_PER_NODE \
#                 training.py \
#                 --model_name ${model_name} \
#                 --grammar_name ${grammar_name} \
#                 --num_samples 10000 \
#                 --num_train_epochs 50 \
#                 --data_seed 5 \
#                 --run_seed ${run_seed} \
#                 --comment ${comment} \
#                 --considered_training_samples ${considered_training_samples} \
#                 --considered_eval_samples 128 \
#                 --store_result \
#                 --use_deepspeed \
#                 --include_incorrect_random_eval \
#                 --include_edit_distance_eval \
#                 --combine_edit_distance \
                
  
            
#             done
#         done 
#     done
# done


