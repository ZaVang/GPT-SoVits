# 使用Miniconda镜像作为基础镜像
FROM continuumio/miniconda3

# 安装依赖项
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    build-essential \
    wget \
    curl \
    cmake \
    vim \
    tmux \
    ffmpeg \
    libglu1-mesa \
    libxi-dev \
    libxmu-dev \
    libglu1-mesa-dev \
    freeglut3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# # 下载CUDA工具包并安装
# RUN wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run \
#     && sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit > /dev/null \
#     && rm cuda_11.8.0_520.61.05_linux.run

# # 更新环境变量
# ENV PATH=$PATH:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/nvidia/lib64
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# 创建并初始化conda环境
RUN conda create -n aispeech python=3.10.13 && \
    echo "conda activate aispeech" >> ~/.bashrc

SHELL ["conda", "run", "-n", "aispeech", "/bin/bash", "-c"]

WORKDIR /app

# 安装依赖

COPY install.sh /app/install.sh
COPY requirements.txt /app/requirements.txt
RUN chmod +x /app/install.sh && bash /app/install.sh -y

# 将当前目录内容复制到容器的/app下
COPY . /app

# 在Dockerfile中设置环境变量
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

