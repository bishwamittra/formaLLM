#!/bin/bash
#
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --gres=gpu:4            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 7-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-8h100grid-01,sws-8h100grid-02,sws-8h100grid-03


#SBATCH -o %x_%j.log      # File to which STDOUT will be written
#SBATCH -e %x_%j.err      # File to which STDERR will be written


# DOCKER_STORAGE_PATH="/NS/venvs/nobackup/bghosh/docker_images/formal_grammar_docker"
# docker load < ${DOCKER_STORAGE_PATH}


# docker run --rm \
# -v /NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training:/home/exp/training \
# -v /NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training/artifacts:/home/exp/training/artifacts \
# -v /NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning/training/todo_and_log:/home/exp/training/todo_and_log \
# -v /NS/llm-1/nobackup/shared/huggingface_cache/hub:/home/exp/.hf_cache/ \
# --shm-size 128gb \
# -e HF_HUB_CACHE=/home/exp/.hf_cache/hub \
# --gpus all \
# -it formal_grammar_docker touch /home/exp/artifacts/dummy.txt

# echo "Start Sleeping"
# sleep 10
# echo "Done Sleeping"


project_dir="/NS/llm-1/nobackup/bishwa/work/formal_grammars"
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
-v /NS/llm-1/nobackup/bishwa/work/formal_grammars/grammar_learning:/home/exp \
-v /NS/llm-1/nobackup/shared/huggingface_cache:/home/exp/.hf_cache \
--shm-size 128gb \
-e HF_HUB_CACHE=/home/exp/.hf_cache/hub \
formal_grammar_docker bash training/todo_and_log/todo_docker.sh