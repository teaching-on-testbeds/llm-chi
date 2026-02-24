FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.12
# builds on https://github.com/jupyter/docker-stacks/blob/main/images/pytorch-notebook/cuda12/Dockerfile

USER root

# nvtop is for monitoring GPU utilization
RUN apt update && apt install nvtop

USER ${NB_UID}

RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools

RUN python -m pip install --no-cache-dir bash_kernel && \
    python -m bash_kernel.install --sys-prefix

# Install Lightning and Transformers
RUN python -m pip install --pre --no-cache-dir lightning transformers datasets accelerate sentencepiece bitsandbytes peft && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"