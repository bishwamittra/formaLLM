#!/bin/bash
#
#SBATCH --partition=h200      # Use GPU partition "a100"
#SBATCH --gres=gpu:4            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 8-00:00              # Maximum run-time in D-HH:MMs
#SBATCH --mem=400GB             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h200grid-01,sws-8h200grid-02,sws-8h200grid-04,sws-8h200grid-05,sws-8h200grid-06


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written

docker rmi $(docker images -a -q)


project_dir="/NS/formal-grammar-and-memorization/work/bghosh/formal_grammars"
cd ${project_dir}

if [ -z "$(docker images -q formal_grammar_docker:latest 2> /dev/null)" ]; then
  echo "formal_grammar_docker:latest not found"
  docker build -t formal_grammar_docker .
else 
  echo "formal_grammar_docker:latest found"
fi

echo "Starting docker"
docker images
docker ps -a



docker run --rm \
--gpus all \
-v /NS/formal-grammar-and-memorization/work/bghosh/formal_grammars/grammar_learning:/home/exp \
-v /NS/formal-grammar-and-memorization/nobackup/shared/huggingface_cache:/home/exp/.hf_cache \
-v /NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/soumi:/home/exp/training/base_models_soumi \
-v /NS/formal-grammar-and-memorization/nobackup/bghosh/temp_models/vnanda:/home/exp/training/base_models_vnanda \
--shm-size 128gb \
-e HF_HUB_CACHE=/home/exp/.hf_cache/hub \
formal_grammar_docker bash training/todo_and_log/run_docker_vary_training_data_size_without_checkpoint_dup_23_todo.sh