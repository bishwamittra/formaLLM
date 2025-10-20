nvidia-smi

export GPUS_PER_NODE=4
cd training











comment="2025_05_01_under_trained_tokens_fine_tuning"






for run_seed in "0"
do

    for model_name in  \
                    "base_models_vnanda/Llama-2-7b-hf" \
                                   

    do



        for considered_training_samples in "16" "64" "256" "1024"
        do



            for grammar_name in "pcfg_cfg3b_disjoint_terminals"
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
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --use_under_trained_tokens



            done
        done
    done
done



for run_seed in "1" "2"
do


    for considered_training_samples in "16" "64" "256" "1024"
    do



        for model_name in  \
                        "base_models_vnanda/Llama-2-7b-hf" \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "EleutherAI/pythia-6.9b" \
                        "mistralai/Mistral-7B-v0.3" \

        do



        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_cfg3b_disjoint_terminals"
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
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --use_under_trained_tokens




            done
        done
    done
done












for run_seed in "0"
do


    for considered_training_samples in "16" "64" "256" "1024"
    do



        for model_name in  \
                        "Qwen/Qwen2.5-7B" \
                        "base_models_vnanda/Llama-2-7b-hf" \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "EleutherAI/pythia-6.9b" \
                        "mistralai/Mistral-7B-v0.3" \

        do



        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
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
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --use_under_trained_tokens




            done
        done
    done
done












for run_seed in "1" "2"
do


    for considered_training_samples in "16" "64" "256" "1024"
    do



        for model_name in  \
                        "Qwen/Qwen2.5-7B" \
                        

        do



        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
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
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --use_under_trained_tokens




            done
        done
    done
done












