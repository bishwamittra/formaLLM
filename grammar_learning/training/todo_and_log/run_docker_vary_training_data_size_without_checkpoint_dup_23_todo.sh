nvidia-smi

export GPUS_PER_NODE=4
cd training




comment="2025_05_12_fine_tuning_g1_grammar_recognize_missing"


for model_name in \
                "base_models_vnanda/Llama-2-13b-hf" \
                


do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals"
    do


            for run_seed in  "0"
            do
                        


                                                              
            for considered_training_samples in "1024"
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





comment="2025_05_12_fine_tuning_g3_g4_grammar_recognize_missing"


for model_name in \
                "base_models_vnanda/Llama-2-13b-hf" \
                "meta-llama/Llama-3.2-1B" \
                "meta-llama/Llama-3.2-3B" \
                "google/gemma-2-2b" \
                "base_models_soumi/opt-model-1.3B" \
                "base_models_soumi/opt-model-2.7B" \
                "base_models_soumi/opt-model-6.7B" \
                "EleutherAI/pythia-1b" \
                "EleutherAI/pythia-2.8b" \
                


do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals_latin" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
    do


            for run_seed in  "0"
            do
                        


                                                              
            for considered_training_samples in "16" "64" "256" "1024"
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






comment="2025_05_12_fine_tuning_g3_g4_grammar_recognize_missing"


for model_name in \
                "meta-llama/Meta-Llama-3.1-8B" \
                


do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals_latin"
    do


            for run_seed in  "0"
            do
                        


                                                              
            for considered_training_samples in "16" "64" "256" "1024"
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



# icl
comment="2025_05_12_long_context_incontext_input_missing"


# Llama-3 and Gemma
for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "256" "512" "1024" "1536"
    do

        for model_name in \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "meta-llama/Llama-3.2-1B" \
                        "meta-llama/Llama-3.2-3B" \
                        "google/gemma-2-2b" \
                        "google/gemma-2-9b" \
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done


# Llama-3
for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1536"
    do

        for model_name in \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "meta-llama/Llama-3.2-1B" \
                        "meta-llama/Llama-3.2-3B" \
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin" "pcfg_cfg3b_disjoint_terminals_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done



for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1024"
    do

        for model_name in \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        "meta-llama/Llama-3.2-3B" \
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_cfg3b_disjoint_terminals_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done



for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1024"
    do

        for model_name in \
                        "meta-llama/Meta-Llama-3.1-8B" \
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done





# mistral
for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "512" "1024" "1536"
    do

        for model_name in \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_cfg3b_disjoint_terminals_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done


for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1024" "1536"
    do

        for model_name in \
                        "mistralai/Mistral-Nemo-Base-2407"\
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done




for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "512" "1024" "1536"
    do

        for model_name in \
                        "mistralai/Mistral-7B-v0.3" \
                        
                        
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done











# Qwen
for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1024" "1536"
    do

        for model_name in \
                        "Qwen/Qwen2.5-0.5B" \
                        "Qwen/Qwen2.5-7B" \
                        
                        
                        
        do

            
            for grammar_name in "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin" "pcfg_cfg3b_disjoint_terminals_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done


for run_seed in "0" "1" "2"
do



    for considered_incontext_examples in "1536"
    do

        for model_name in \
                        "Qwen/Qwen2.5-1.5B" \
                        
                        
                        
        do

            
            for grammar_name in  "pcfg_4_3_1_2_3_4_5_6_7_8_9" "pcfg_4_3_1_2_3_4_5_6_7_8_9_latin" "pcfg_cfg3b_disjoint_terminals_latin"
                              

                              
            do
                time TRANSFORMERS_VERBOSITY=error python \
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
                --icl_batch_size 2


            done
        done
    done
done


