grammar_goodname = {
    "pcfg_ab" : "(ab)+",
    "pcfg_ab_aabb": "(ab)+ | (aabb)+",
    "pcfg_ab_aabb_mixed": "(aa | aabb)+",
    "pcfg_cfg3b": "cfg3b",
    "pcfg_one_character_missing": "(a|b)+ | (b|c)+ | (a|c)+",
    "pcfg_cfg3b_disjoint_terminals": "cfg3b balanced",
    "pcfg_cfg3b_disjoint_terminals_skewed_prob": "cfg3b skewed",
    "pcfg_balanced_parenthesis": "balanced parenthesis",
    "pcfg_reverse_string": "reverse string",
    "pcfg_cfg3b_disjoint_terminals_combined": "cfg3b balanced",
    "pcfg_cfg3b_disjoint_terminals_skewed_prob_combined": "cfg3b skewed",
    "pcfg_cfg3b_disjoint_terminals_one_rule_missing": "cfg3b one rule missing",
    "pcfg_cfg3b_eq_len_skewed_prob": "skewed 0.95",
    "pcfg_cfg3b_eq_len_skewed_prob_0.75": "skewed 0.75",
    "pcfg_cfg3b_eq_len_uniform_prob": "balanced",
}


model_revised_location = {
    "base_models_soumi/opt-model-1.3B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-1.3B",
    "base_models_soumi/opt-model-2.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B",
    "base_models_soumi/opt-model-6.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-6.7B",

    "base_models_soumi//opt-model-1.3B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-1.3B",
    "base_models_soumi//opt-model-2.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B",
    "base_models_soumi//opt-model-6.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-6.7B",

    "/NS/llm-1/nobackup/soumi/opt-model-1.3B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-1.3B",
    "/NS/llm-1/nobackup/soumi/opt-model-2.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B",
    "/NS/llm-1/nobackup/soumi/opt-model-6.7B": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-6.7B",


    "base_models_vnanda/Llama-2-7b-hf": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-7b-hf",
    "base_models_vnanda/Llama-2-13b-hf": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-13b-hf",

    "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-7b-hf",
    "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf": "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-13b-hf",


}

def get_eval_dataset_goodname(eval_datasets):
    eval_dataset_goodname = {}
    for eval_dataset in eval_datasets:
        if(eval_dataset == 'test_sequences'):
            eval_dataset_goodname[eval_dataset] = 'Correct Test'
        elif(eval_dataset == 'train_sequences'):
            eval_dataset_goodname[eval_dataset] = 'Train'
        elif(eval_dataset == 'non_grammatical_sequences'):
            eval_dataset_goodname[eval_dataset] = 'Incorrect Random'
            # eval_dataset_goodname[eval_dataset] = 'Test Random'
        else:
            original_eval_dataset = eval_dataset
            # eval_dataset = eval_dataset.replace('non_grammatical_train', 'Incorrect Train ')
            eval_dataset = eval_dataset.replace('non_grammatical_test', 'Incorrect')
            eval_dataset = eval_dataset.replace('_sequences_edit_distance_', ' by ')
            eval_dataset = eval_dataset.replace('_sequences_grammar_edit_', 'Edit ')
            eval_dataset = eval_dataset.replace('_sequences_bucket_', 'Len: ')
            eval_dataset = eval_dataset.replace('subgrammar_sequences_', 'Rule: ')
            eval_dataset = eval_dataset.replace('test_sequences_all_rules', 'Test All Rules')

            # # eval_dataset = eval_dataset.replace('non_grammatical_train', 'Incorrect Train ')
            # eval_dataset = eval_dataset.replace('non_grammatical_test', 'Test')
            # eval_dataset = eval_dataset.replace('_sequences_edit_distance_', ' ')
            # eval_dataset = eval_dataset.replace('_sequences_grammar_edit_', 'Edit ')
            # eval_dataset = eval_dataset.replace('_sequences_bucket_', 'Len: ')
            # eval_dataset = eval_dataset.replace('subgrammar_sequences_', 'Rule: ')
            # eval_dataset = eval_dataset.replace('test_sequences_all_rules', 'Test All Rules')

            if "edit" in original_eval_dataset:
                eval_dataset += " Edit"
            eval_dataset_goodname[original_eval_dataset] = eval_dataset
    
    return eval_dataset_goodname

model_goodname = {
    "llama2-7b": "Llama-2-7B",
    "gpt2-large": "GPT-2 Large",
    "mistral-7b": "Mistral-7B",
    "pythia-6.9b": "Pythia-6.9B",
    "llama3-8b": "Llama-3-8B",

    
    "Qwen/Qwen2.5-0.5B": "Qwen-2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen-2.5-1.5B",
    "Qwen/Qwen2.5-7B": "Qwen-2.5-7B",
    "Qwen/Qwen2.5-14B": "Qwen-2.5-14B",
    
    'meta-llama/Meta-Llama-3-8B': 'Llama-3-8B',
    'meta-llama/Meta-Llama-3.1-8B': 'Llama-3.1-8B',
    'meta-llama/Llama-3.2-3B': 'Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B': 'Llama-3.2-1B',
    
    'EleutherAI/pythia-6.9b': 'Pythia-6.9B',
    'EleutherAI/pythia-1b': 'Pythia-1B',
    'EleutherAI/pythia-2.8b': 'Pythia-2.8B',
    
    'mistralai/Mistral-7B-v0.3': 'Mistral-7B',
    'mistralai/Mistral-Nemo-Base-2407': 'Mistral-12B',

    "google/gemma-2-2b": "Gemma-2-2B",
    "google/gemma-2-9b": "Gemma-2-9B",

    "/NS/llm-1/nobackup/soumi/opt-model-1.3B": "Opt-1.3B",
    "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-1.3B": "Opt-1.3B",
    "base_models_soumi/opt-model-1.3B": "Opt-1.3B",
    "/NS/llm-1/nobackup/soumi/opt-model-2.7B": "Opt-2.7B",
    "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-2.7B": "Opt-2.7B",
    "base_models_soumi/opt-model-2.7B": "Opt-2.7B",
    "/NS/llm-1/nobackup/soumi/opt-model-6.7B": "Opt-6.7B",
    "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi/opt-model-6.7B": "Opt-6.7B",
    "base_models_soumi/opt-model-6.7B": "Opt-6.7B",

    "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-7b-hf": "Llama-2-7B",
    "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-7b-hf": "Llama-2-7B",
    "/NS/llm-1/nobackup/vnanda/llm_base_models/Llama-2-13b-hf": "Llama-2-13B",
    "/NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda/Llama-2-13b-hf": "Llama-2-13B",
    "base_models_vnanda/Llama-2-13b-hf": "Llama-2-13B",
}

