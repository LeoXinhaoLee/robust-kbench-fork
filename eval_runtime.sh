#!/usr/bin/env bash

set -euo pipefail

set -x

linux_platform=$(uname -m)
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CUDA_HOME
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/${linux_platform}-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/${linux_platform}-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/${linux_platform}-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/${linux_platform}-linux/lib:${LD_LIBRARY_PATH:-}"


## EOS need
export CUDAHOSTCXX=$(which g++)
export CXX=$CUDAHOSTCXX
export CC=$(which gcc)
##


node=$(hostname)

# tasks=(mnist_linear_relu mnist_cross_entropy resnet_block)
# modes=(backward forward forward)
# tasks=(resnet_block)
# modes=(forward)

tasks=(layernorm llama_ffw llama_rmsnorm mnist_conv_relu_pool mnist_linear mnist_linear_relu mnist_cross_entropy mnist_linear mnist_pool)
modes=(forward forward forward forward forward forward backward backward backward)


for i in "${!tasks[@]}"; do

  task="${tasks[$i]}"
  mode="${modes[$i]}"

  for i in {0..5}; do
      echo "Run $i"

      if [[ ${mode} == "backward" ]]; then
        python run_kernel.py \
          --task_dir tasks/${task} \
          --cuda_code_path highlighted/${task}/${mode}/kernel.cu \
          --backward \
          --temp_suffix "_${node}"
      else
        python run_kernel.py \
          --task_dir tasks/${task} \
          --cuda_code_path highlighted/${task}/${mode}/kernel.cu \
          --temp_suffix "_${node}"
      fi

      mkdir -p tasks/${task}/${mode}/${node}
      mv tasks/${task}/${mode}/eval_results_${node} \
         tasks/${task}/${mode}/${node}/eval_results_${i}

      mkdir -p highlighted/${task}/${mode}/${node}
      mv highlighted/${task}/${mode}/eval_results_${node} \
         highlighted/${task}/${mode}/${node}/eval_results_${i}

  done    

done


# tgt_dir="TTT/mnist_linear_relu/backward"

# for cu_file in "${tgt_dir}"/step_*.cu; do
#   # Skip if glob didn't match
#   [[ -e "$cu_file" ]] || continue

#   # Get filename only: step_XX.cu
#   base="$(basename "$cu_file")"

#   # Extract step number
#   step="${base#step_}"
#   step="${step%.cu}"

#   save_dir="${tgt_dir}/step_${step}"
#   mkdir -p "$save_dir"

#   for i in {1..5}; do
#     echo "Step ${step} — Run $i"

#     python run_kernel.py \
#       --task_dir tasks/mnist_linear_relu \
#       --cuda_code_path "$cu_file" \
#       --backward

#     mv "${tgt_dir}/eval_results" \
#        "${save_dir}/eval_results_${i}"
#   done
# done
