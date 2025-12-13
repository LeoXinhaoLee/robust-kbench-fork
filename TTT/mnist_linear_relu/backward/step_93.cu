#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_dz_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    int batch_size,
    int inF,
    int outF,
    float* __restrict__ dz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size * outF) return;

    int b = tid / outF;
    int o = tid % outF;

    float accum = 0.0f;
    for (int i = 0; i < inF; ++i) {
        int x_idx = b * inF + i;
        int w_idx = o * inF + i;
        accum += x[x_idx] * weights[w_idx];
    }
    accum += biases[o];
    float z = accum;
    float mask = (z > 0.0f) ? 1.0f : 0.0f;
    dz[tid] = grad_output[tid] * mask;
}

__global__ void compute_grad_input(
    const float* __restrict__ dz,
    const float* __restrict__ weights,
    int batch_size,
    int inF,
    int outF,
    float* __restrict__ grad_input) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size * inF) return;

    int b = tid / inF;
    int i = tid % inF;

    float acc = 0.0f;
    for (int o = 0; o < outF; ++o) {
        int dz_idx = b * outF + o;
        int w_idx = o * inF + i;
        acc += dz[dz_idx] * weights[w_idx];
    }
    grad_input[tid] = acc;
}

__global__ void compute_grad_weights(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    int batch_size,
    int inF,
    int outF,
    float* __restrict__ grad_weights) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= outF * inF) return;

    int o = tid / inF;
    int i = tid % inF;

    float acc = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int dz_idx = b * outF + o;
        int x_idx = b * inF + i;
        acc += dz[dz_idx] * x[x_idx];
    }
    grad_weights[tid] = acc;
}

__global__ void compute_grad_biases(
    const float* __restrict__ dz,
    int batch_size,
    int outF,
    float* __restrict__ grad_biases) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= outF) return;

    float acc = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int dz_idx = b * outF + tid;
        acc += dz[dz_idx];
    }
    grad_biases[tid] = acc;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor biases) {
    int batch_size = x.size(0);
    int inF = x.size(1);
    int outF = weights.size(0);

    auto grad_input = torch::empty({batch_size, inF}, x.options());
    auto grad_weights = torch::empty({outF, inF}, weights.options());
    auto grad_biases = torch::empty({outF}, biases.options());

    auto dz_tensor = torch::empty({batch_size * outF}, x.options());
    float* dz = dz_tensor.data_ptr<float>();

    int threads_per_block = 256;
    int total_threads = batch_size * outF;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    compute_dz_kernel<<<num_blocks, threads_per_block>>>(
        grad_output.data<float>(),
        x.data<float>(),
        weights.data<float>(),
        biases.data<float>(),
        batch_size, inF, outF, dz);

    total_threads = batch_size * inF;
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    compute_grad_input<<<num_blocks, threads_per_block>>>(
        dz, weights.data<float>(), batch_size, inF, outF, grad_input.data<float>());

    total_threads = outF * inF;
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    compute_grad_weights<<<num_blocks, threads_per_block>>>(
        dz, x.data<float>(), batch_size, inF, outF, grad_weights.data<float>());

    total_threads = outF;
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    compute_grad_biases<<<num_blocks, threads_per_block>>>(
        dz, batch_size, outF, grad_biases.data<float>());

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}