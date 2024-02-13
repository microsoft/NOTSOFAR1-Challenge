# Base
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest as base

# The base image comes with Python 3.10.13 preinstalled. Hence, no need to install it here.

# Argument for specifying requirements file, default is "requirements.txt"
ARG REQUIREMENTS_FILE=requirements.txt

# Libs
RUN apt-get -y update && \
    apt-get install -y software-properties-common ffmpeg libportaudio2 \
    libasound-dev git git-lfs zlib1g-dev libreadline-dev \
    libncursesw5-dev libnss3-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    libsndfile1 libsnappy-dev openmpi-bin graphviz libsm6 libxext6 libxrender-dev make build-essential \
    curl llvm libncurses5-dev xz-utils libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb
RUN dpkg -i libffi7_3.3-4_amd64.deb

# A few tools to aid debugging
RUN apt-get -y install smem dstat man less screen
RUN pip install ps_mem ipython

# Virtualenv
RUN pip install virtualenv

# dependencies for running the inference pipeline packages
RUN python -m pip install --upgrade pip
RUN pip install --upgrade setuptools wheel Cython fasttext-wheel
RUN apt-get install python3.10-dev ffmpeg build-essential

# Packages
ARG CACHE_BUST
COPY *requirements.txt /home/
RUN pip install -r /home/${REQUIREMENTS_FILE}