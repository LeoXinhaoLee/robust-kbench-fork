#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_dz_kernel(
    const float* x,
    const float* weights,
    const float* biases,
    const float* grad_output,
    float* dz,
    int batch_size,
    int in_features,
    int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int b = idx / out_features;
    int i = idx % out_features;

    float sum = 0.0f;
    int x_start = b * in_features;
    int w_start = i * in_features;
    for (int j = 0; j < in_features; ++j) {
        sum += x[x_start + j] * weights[w_start + j];
    }
    float z = sum + biases[i];
    dz[idx] = (z > 0.0f) ? grad_output[idx] : 0.0f;
}

__global__ void compute_grad_weights_kernel(
    const float* dz,
    const float* x,
    float* grad_weights,
    int batch_size,
    int in_features,
    int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_features * in_features) return;

    int i = idx / in_features;
    int j = idx % in_features;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int dz_idx = b * out_features + i;
        int x_idx = b * in_features + j;
        sum += dz[dz_idx] * x[x_idx];
    }
    grad_weights[i * in_features + j] = sum;
}

__global__ void compute_grad_input_kernel(
    const float* dz,
    const float* weights,
    float* grad_input,
    int batch_size,
    int in_features,
    int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_features) return;

    int b = idx / in_features;
    int j = idx % in_features;

    float sum = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        int dz_idx = b * out_features + i;
        int w_idx = i * in_features + j;
        sum += dz[dz_idx] * weights[w_idx];
    }
    grad_input[b * in_features + j] = sum;
}

__global__ void compute_grad_biases_kernel(
    const float* dz,
    float* grad_biases,
    int batch_size,
    int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int idx = b * out_features + i;
        sum += dz[idx];
    }
    grad_biases[i] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor weights, 
    torch::Tensor biases) {

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weights.size(0); 

    auto options = x.options();
    torch::Tensor dz = torch::empty({batch_size, out_features}, options);
    torch::Tensor grad_weights = torch::empty({out_features, in_features}, options);
    torch::Tensor grad_input = torch::empty({batch_size, in_features}, options);
    torch::Tensor grad_biases = torch::empty({out_features}, options);

    int threads_per_block = 256;
    int total_threads_dz = batch_size * out_features;
    int blocks_dz = (total_threads_dz + threads_per_block - 1) / threads_per_block;
    compute_dz_kernel<<<blocks_dz, threads_per_block>>>(
        x.data<float>(), weights.data<float>(), biases.data<float>(), 
        grad_output.data<float>(), dz.data<float>(),
        batch_size, in_features, out_features);

    int total_threads_weights = out_features * in_features;
    int blocks_weights = (total_threads_weights + threads_per_block - 1) / threads_per_block;
    compute_grad_weights_kernel<<<blocks_weights, threads_per_block>>>(
        dz.data<float>(), x.data<float>(), grad_weights.data<float>(), 
        batch_size, in_features, out_features);

    int total_threads_input = batch_size * in_features;
    int blocks_input = (total_threads_input + threads_per_block - 1) / threads_per_block;
    compute_grad_input_kernel<<<blocks_input, threads_per_block>>>(
        dz.data<float>(), weights.data<float>(), grad_input.data<float>(),
        batch_size, in_features, out_features);

    int total_threads_biases = out_features;
    int blocks_biases = (total_threads_biases + threads_per_block - 1) / threads_per_block;
    compute_grad_biases_kernel<<<blocks_biases, threads_per_block>>>(
        dz.data<float>(), grad_biases.data<float>(), 
        batch_size, out_features);

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}