#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute grad_relu
__global__ void compute_grad_relu_kernel(
    const float* x, 
    const float* weights, 
    const float* biases, 
    const float* grad_output,
    float* grad_relu,
    int batch_size, 
    int input_dim, 
    int output_dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size * output_dim) return;

    int b = tid / output_dim;
    int j = tid % output_dim;

    float sum = 0.0f;
    for (int i = 0; i < input_dim; ++i) {
        sum += x[b * input_dim + i] * weights[j * input_dim + i];
    }
    float y = sum + biases[j];
    float mask = (y > 0.0f) ? 1.0f : 0.0f;

    int grad_relu_idx = b * output_dim + j;
    grad_relu[grad_relu_idx] = grad_output[grad_relu_idx] * mask;
}

__global__ void compute_grad_input_kernel(
    const float* grad_relu,
    const float* weights,
    float* grad_input,
    int batch_size,
    int input_dim,
    int output_dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size * input_dim) return;

    int b = tid / input_dim;
    int i = tid % input_dim;

    float sum = 0.0f;
    for (int j = 0; j < output_dim; ++j) {
        sum += grad_relu[b * output_dim + j] * weights[j * input_dim + i];
    }

    grad_input[b * input_dim + i] = sum;
}

__global__ void compute_grad_weights_kernel(
    const float* grad_relu,
    const float* x,
    float* grad_weights,
    int batch_size,
    int input_dim,
    int output_dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= output_dim * input_dim) return;

    int j = tid / input_dim;
    int i = tid % input_dim;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += grad_relu[b * output_dim + j] * x[b * input_dim + i];
    }

    grad_weights[j * input_dim + i] = sum;
}

__global__ void compute_grad_biases_kernel(
    const float* grad_relu,
    float* grad_biases,
    int batch_size,
    int output_dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= output_dim) return;

    int j = tid;
    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += grad_relu[b * output_dim + j];
    }

    grad_biases[j] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor biases
) {
    int batch_size = x.size(0);
    int input_dim = x.size(1);
    int output_dim = weights.size(0);

    auto options = x.options();
    auto grad_input = torch::empty({batch_size, input_dim}, options);
    auto grad_weights = torch::empty({output_dim, input_dim}, options);
    auto grad_biases = torch::empty({output_dim}, options);
    auto grad_relu = torch::empty({batch_size, output_dim}, options);

    int threads_per_block = 256;
    int total_threads_grad_relu = batch_size * output_dim;
    int num_blocks = (total_threads_grad_relu + threads_per_block - 1) / threads_per_block;
    compute_grad_relu_kernel<<<num_blocks, threads_per_block>>>(
        x.data<float>(), weights.data<float>(), biases.data<float>(),
        grad_output.data<float>(), grad_relu.data<float>(),
        batch_size, input_dim, output_dim);

    int total_threads_grad_input = batch_size * input_dim;
    num_blocks = (total_threads_grad_input + threads_per_block - 1) / threads_per_block;
    compute_grad_input_kernel<<<num_blocks, threads_per_block>>>(
        grad_relu.data<float>(), weights.data<float>(),
        grad_input.data<float>(),
        batch_size, input_dim, output_dim);

    int total_threads_grad_weights = output_dim * input_dim;
    num_blocks = (total_threads_grad_weights + threads_per_block - 1) / threads_per_block;
    compute_grad_weights_kernel<<<num_blocks, threads_per_block>>>(
        grad_relu.data<float>(), x.data<float>(),
        grad_weights.data<float>(),
        batch_size, input_dim, output_dim);

    int total_threads_grad_biases = output_dim;
    num_blocks = (total_threads_grad_biases + threads_per_block - 1) / threads_per_block;
    compute_grad_biases_kernel<<<num_blocks, threads_per_block>>>(
        grad_relu.data<float>(),
        grad_biases.data<float>(),
        batch_size, output_dim);

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}