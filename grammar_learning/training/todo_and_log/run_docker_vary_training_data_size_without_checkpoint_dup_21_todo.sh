nvidia-smi

export GPUS_PER_NODE=4
cd training





comment="2025_05_05_more_test_data_memorization_fine_tuning"





# Mistral 7B
for run_seed in "1"
do

    for considered_training_samples in "16"
    do
    
    
        for model_name in  \
                        "mistralai/Mistral-7B-v0.3"

                        
                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done











for run_seed in "2"
do

    for considered_training_samples in "16" "64"
    do
    
    
        for model_name in  \
                        "mistralai/Mistral-7B-v0.3"
                        
                        
                        

        do

        



            for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done






for run_seed in "2"
do

    for considered_training_samples in "256" "1024"
    do
    
    
        for model_name in  \
                        "mistralai/Mistral-7B-v0.3"
                        
                        
                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done








# pythia
for run_seed in "1"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "EleutherAI/pythia-6.9b" \
                        
                        
                        

        do

        



            for grammar_name in "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done




for run_seed in "2"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "EleutherAI/pythia-6.9b" \
                        "google/gemma-2-9b" \
                        
                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done



# gemma
for run_seed in "1"
do

    for considered_training_samples in "1024"
    do
    
    
        for model_name in  \
                        "google/gemma-2-9b" \
                        
                        
                        

        do

        



            for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done







# rest
for run_seed in "0"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "EleutherAI/pythia-2.8b" \
                        "meta-llama/Llama-3.2-3B" \
                        "base_models_soumi/opt-model-2.7B" \

                        
                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done





for run_seed in "1" "2"
do

    for considered_training_samples in "16" "64" "256" "1024"
    do
    
    
        for model_name in  \
                        "Qwen/Qwen2.5-1.5B" \
                        "EleutherAI/pythia-2.8b" \
                        "google/gemma-2-2b" \
                        "meta-llama/Llama-3.2-1B" \
                        "meta-llama/Llama-3.2-3B" \
                        "base_models_soumi/opt-model-1.3B" \
                        "base_models_soumi/opt-model-2.7B" \
                        
                        
                        

        do

        



            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_skewed_prob" "pcfg_4_3_1_2_3_4_5_6_7_8_9_eq_len_uniform_prob" "pcfg_cfg3b_eq_len_skewed_prob" "pcfg_cfg3b_eq_len_uniform_prob"
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
                --considered_eval_samples 1024 \
                --store_result \
                --use_deepspeed


                # --include_incorrect_random_eval \
                # --include_edit_distance_eval \
                # --combine_edit_distance \
                


                
                

            done
        done
    done
done









