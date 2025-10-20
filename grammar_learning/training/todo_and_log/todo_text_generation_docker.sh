export GPUS_PER_NODE=4
workdir="training"
cd $workdir

nvidia-smi

comment="Week_2024.11.03_text_generation_callback"

for grammar_name in "pcfg_cfg3b_eq_len_uniform_prob"
do
    for considered_training_samples in "64"
    do
        for run_seed in "0" "1" "2" "3" "4"
        do
            for model_name in "base_models_soumi/opt-model-1.3B" \
                            #   "mistralai/Mistral-Nemo-Base-2407" \
                            #   "meta-llama/Meta-Llama-3-8B" \
                                

            do
                time torchrun --nproc_per_node=$GPUS_PER_NODE \
                training.py \
                --model_name ${model_name} \
                --grammar_name ${grammar_name} \
                --num_samples 10000 \
                --num_train_epochs 100 \
                --data_seed 5 \
                --run_seed ${run_seed} \
                --comment ${comment} \
                --considered_training_samples ${considered_training_samples} \
                --considered_eval_samples ${considered_training_samples} \
                --store_result \
                --max_new_tokens 20 \
                --use_deepspeed \
                --generate_text \


            done
        done
    done
done

