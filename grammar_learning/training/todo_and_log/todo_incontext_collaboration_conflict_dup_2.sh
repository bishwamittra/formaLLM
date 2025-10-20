#!/bin/bash
#
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres=gpu:4          # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-07,sws-8h100grid-05



#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written
export GPUS_PER_NODE=4
export HF_HUB_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/hub
export HF_DATASETS_CACHE=/NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache/datasets

workdir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning/training"
cd $workdir
nvidia-smi

                    
