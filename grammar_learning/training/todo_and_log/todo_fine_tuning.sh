
nvidia-smi

export GPUS_PER_NODE=4
cd training


comment="2024_10_17_save_checkpoint"
for grammar_name in "pcfg_cfg3b_disjoint_terminals"
do
    for considered_training_samples in "64" "128" "256" "512" "1024"
    do
        for run_seed in "0" "1" "2"
        do
            for model_name in "mistralai/Mistral-7B-v0.3" \
                        

            do
                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
                training_pcfg.py \
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
                --use_deepspeed \
                --include_incorrect_random_eval \
                --include_edit_distance_eval \
                --combine_edit_distance \
                --save_best_model \
                
  
            
            done
        done 
    done
done