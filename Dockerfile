FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    git \
    wget 

RUN apt install -y bzip2 \
    ca-certificates

RUN DEBIAN_FRONTEND=noninteractive apt install -y libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1\
    mercurial \
    subversion \ 
    libgl1-mesa-glx

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda clean -tipy

RUN conda create --name cis6200 python=3.11

RUN echo "source activate cis6200" >> ~/.bashrc
ENV PATH /opt/conda/envs/camp_zipnerf/bin:$PATH
RUN conda install pip

COPY . /cis6200
WORKDIR /cis6200

# Clone the diffusers repository
RUN git clone https://github.com/huggingface/diffusers.git

# Change to the diffusers directory
WORKDIR /cis6200/diffusers

RUN pip install --upgrade pip && \
    pip install -e .

WORKDIR /cis6200/diffusers/examples/controlnet
RUN pip install -r requirements.txt

WORKDIR /cis6200

# Create a basic configuration for accelerate
RUN python -c "from accelerate.utils import write_basic_config; write_basic_config()"


