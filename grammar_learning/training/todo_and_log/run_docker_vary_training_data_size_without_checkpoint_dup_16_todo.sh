nvidia-smi

export GPUS_PER_NODE=4
cd training


comment="2025_08_07_save_checkpoint_block_structure_fine_tuning"


for run_seed in "0"
do


    for grammar_name in "pcfg_cfg3b_disjoint_terminals"
    do


    
        

        for model_name in "Qwen/Qwen2.5-7B" \
        # for model_name in "EleutherAI/pythia-1b" \


        do                                                      


            
            for considered_training_samples in "64"
            do


                time torchrun \
                --nproc_per_node=$GPUS_PER_NODE \
                training.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 100 \
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
                --save_checkpoint
  
            
            done
        done 
    done
done



