ARG BASE_IMAGE_NAME="docker.io/pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel"
FROM ${BASE_IMAGE_NAME}

# Create the directory for the code
ARG EXP_DIR="/home/exp"
RUN mkdir -p ${EXP_DIR}
WORKDIR $EXP_DIR

# Copy and install dependencies
COPY grammar_learning $EXP_DIR/

RUN pip install --no-cache-dir -r training/requirements310.in