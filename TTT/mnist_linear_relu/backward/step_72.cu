#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_dz_kernel(
    const float* x_data,
    const float* weights_data,
    const float* biases_data,
    const float* grad_output_data,
    float* dz_data,
    int B,
    int D_in,
    int D_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_out) return;

    int i = idx / D_out;
    int k = idx % D_out;

    float sum_z = 0.0f;
    for (int j = 0; j < D_in; ++j) {
        sum_z += x_data[i * D_in + j] * weights_data[k * D_in + j];
    }
    float z = sum_z + biases_data[k];
    float mask = (z > 0.0f) ? 1.0f : 0.0f;
    dz_data[i * D_out + k] = grad_output_data[i * D_out + k] * mask;
}

__global__ void compute_grad_x_kernel(
    const float* dz_data,
    const float* weights_data,
    float* grad_x_data,
    int B,
    int D_in,
    int D_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_in) return;

    int i = idx / D_in;
    int j = idx % D_in;

    float sum = 0.0f;
    for (int k = 0; k < D_out; ++k) {
        sum += dz_data[i * D_out + k] * weights_data[k * D_in + j];
    }
    grad_x_data[i * D_in + j] = sum;
}

__global__ void compute_grad_weights_kernel(
    const float* dz_data,
    const float* x_data,
    float* grad_weights_data,
    int B,
    int D_in,
    int D_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D_out * D_in) return;

    int k = idx / D_in;
    int j = idx % D_in;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        sum += dz_data[i * D_out + k] * x_data[i * D_in + j];
    }
    grad_weights_data[k * D_in + j] = sum;
}

__global__ void compute_grad_biases_kernel(
    const float* dz_data,
    float* grad_biases_data,
    int B,
    int D_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= D_out) return;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        sum += dz_data[i * D_out + k];
    }
    grad_biases_data[k] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor weights, 
    torch::Tensor biases) {
    int B = x.size(0);
    int D_in = x.size(1);
    int D_out = weights.size(0);

    auto options = x.options();
    torch::Tensor grad_x = torch::empty({B, D_in}, options);
    torch::Tensor grad_weights = torch::empty({D_out, D_in}, options);
    torch::Tensor grad_biases = torch::empty({D_out}, options);
    torch::Tensor dz = torch::empty({B, D_out}, options);

    int threads_per_block = 256;
    int dz_blocks = (B * D_out + threads_per_block - 1) / threads_per_block;
    compute_dz_kernel<<<dz_blocks, threads_per_block>>>(
        x.data<float>(), weights.data<float>(), biases.data<float>(),
        grad_output.data<float>(), dz.data<float>(),
        B, D_in, D_out);

    int grad_x_blocks = (B * D_in + threads_per_block - 1) / threads_per_block;
    compute_grad_x_kernel<<<grad_x_blocks, threads_per_block>>>(
        dz.data<float>(), weights.data<float>(), grad_x.data<float>(),
        B, D_in, D_out);

    int grad_w_blocks = (D_out * D_in + threads_per_block - 1) / threads_per_block;
    compute_grad_weights_kernel<<<grad_w_blocks, threads_per_block>>>(
        dz.data<float>(), x.data<float>(), grad_weights.data<float>(),
        B, D_in, D_out);

    int grad_b_blocks = (D_out + threads_per_block - 1) / threads_per_block;
    compute_grad_biases_kernel<<<grad_b_blocks, threads_per_block>>>(
        dz.data<float>(), grad_biases.data<float>(), B, D_out);

    return {grad_x, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}