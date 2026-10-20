# config.py — experiment configuration: model-specific defaults and project settings.
# Customize these values for your setup.

# Models that require CPU offloading with DeepSpeed
model_offloading = [
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-Nemo-Base-2407",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-2-9b",
    "Qwen/Qwen2.5-14B",
    # add local model paths that require CPU offloading here
]


# Per-model learning rates
lr_dict = {
    "EleutherAI/pythia-6.9b": 0.00001,
    "EleutherAI/pythia-1b": 0.00001,
    "EleutherAI/pythia-2.8b": 0.00001,

    "mistralai/Mistral-7B-v0.3": 0.000005,
    "mistralai/Mistral-Nemo-Base-2407": 0.000005,

    "meta-llama/Meta-Llama-3-8B": 0.00005,
    "meta-llama/Meta-Llama-3.1-8B": 0.00005,
    "meta-llama/Llama-3.2-1B": 0.00005,
    "meta-llama/Llama-3.2-3B": 0.00005,

    "google/gemma-2-2b": 0.00005,
    "google/gemma-2-9b": 0.00005,

    "Qwen/Qwen2.5-0.5B": 0.00005,
    "Qwen/Qwen2.5-1.5B": 0.00005,
    "Qwen/Qwen2.5-7B": 0.00005,
    "Qwen/Qwen2.5-14B": 0.00005,

    # add local model paths and their learning rates here

    "/path/to/llm_base_models/Llama-2-7b-hf": 0.000005,
    "/path/to/llm_base_models/Llama-2-13b-hf": 0.000005,

    "/path/to/opt-model-1.3B": 0.000005,
    "/path/to/opt-model-2.7B": 0.000005,
    "/path/to/opt-model-6.7B": 0.000005,

    
}

# Per-model batch sizes
batch_size_dict = {
    "EleutherAI/pythia-6.9b": 8,
    "EleutherAI/pythia-1b": 8,
    "EleutherAI/pythia-2.8b": 8,

    "mistralai/Mistral-7B-v0.3": 8,
    "mistralai/Mistral-Nemo-Base-2407": 8,

    "meta-llama/Meta-Llama-3-8B": 8,
    "meta-llama/Meta-Llama-3.1-8B": 8,
    "meta-llama/Llama-3.2-1B": 8,
    "meta-llama/Llama-3.2-3B": 8,

    "google/gemma-2-2b": 8,
    "google/gemma-2-9b": 8,

    "Qwen/Qwen2.5-14B": 4,

    # add local model paths and their batch sizes here

    "/path/to/llm_base_models/Llama-2-7b-hf": 8,
    "/path/to/llm_base_models/Llama-2-13b-hf": 8,


    "/path/to/opt-model-1.3B": 8,
    "/path/to/opt-model-2.7B": 8,
    "/path/to/opt-model-6.7B": 8,

}