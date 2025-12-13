#!/bin/bash

conda install -y \
  pytorch==2.5.1=py3.11_cuda12.4_cudnn9.1.0_0 \
  pytorch-cuda==12.4=hc786d27_7 \
  -c pytorch -c nvidia -c conda-forge

conda install -y \
  libcufile==1.13.0.11=0 \
  libcurand==10.3.9.55=0 \
  -c nvidia

conda install -y -c nvidia \
  cuda-nvcc=12.4 \
  cuda-nvcc-dev=12.4 \
  cuda-cudart-dev=12.4

conda install -y -c nvidia \
  cuda-toolkit=12.4

conda install -c conda-forge cudnn=8.9.7

pip install -e .

pip install numpy==2.2.6 scipy==1.15.3 sympy==1.13.1 pandas==2.3.3 wandb==0.23.0
