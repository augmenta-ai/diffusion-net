FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS devcontainer

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

COPY environment.yml /tmp/environment.yml
RUN conda env create --name diffusion_net --file /tmp/environment.yml \
    && conda clean --all --yes

ENV PATH="/opt/conda/envs/diffusion_net/bin:${PATH}"
ENV PYTHONPATH="/workspaces/diffusion-net/src"
WORKDIR /workspaces/diffusion-net

FROM devcontainer AS main

COPY . /workspaces/diffusion-net

ENTRYPOINT ["python", "/workspaces/diffusion-net/experiments/classification_agmt/main.py"]