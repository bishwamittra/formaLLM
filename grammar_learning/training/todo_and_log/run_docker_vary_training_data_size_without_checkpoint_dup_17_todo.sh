nvidia-smi

export GPUS_PER_NODE=4
cd training

comment="2025_04_17_long_context_incontext_input"


for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1" "1536" "2048" "2560" "3072" "3584" "4096"
    do

        for model_name in "Qwen/Qwen2.5-0.5B" \
                          "Qwen/Qwen2.5-1.5B" \
                          "Qwen/Qwen2.5-7B" \
                          "Qwen/Qwen2.5-14B" \
                          "google/gemma-2-2b" \
                        "google/gemma-2-9b" \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        "mistralai/Mistral-7B-v0.3" \
                        "EleutherAI/pythia-6.9b" \
                        "EleutherAI/pythia-1b" \
                        "EleutherAI/pythia-2.8b" \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "meta-llama/Llama-3.2-1B" \
                        "meta-llama/Llama-3.2-3B" \
                        "base_models_soumi//opt-model-1.3B" \
                        "base_models_soumi//opt-model-2.7B" \
                        "base_models_soumi//opt-model-6.7B" \
                        "base_models_vnanda/Llama-2-7b-hf" \
                        "base_models_vnanda/Llama-2-13b-hf" \




        do

            
            # for grammar_name in "pcfg_cfg3b_disjoint_terminals" "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_cfg3b_disjoint_terminals_latin" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
            for grammar_name in "pcfg_cfg3b_disjoint_terminals"
                              

                              
            do
                time torchrun --nproc_per_node=$GPUS_PER_NODE \
                training.py \
                --inference_only_mode \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --store_result \
                --incontext_input \
                --considered_training_samples ${considered_incontext_examples} \
                --considered_incontext_examples ${considered_incontext_examples} \
                --num_train_epochs 1 \
                --considered_eval_samples 128 \
                --data_seed 5 \
                --run_seed ${run_seed} \
                --batch_size 1 \
                --comment ${comment} \
                --include_edit_distance_eval \
                --include_incorrect_random_eval \
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
