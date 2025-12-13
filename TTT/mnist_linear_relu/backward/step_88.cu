#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to compute grad_relu (from grad_output, with ReLU mask)
__global__ void compute_grad_relu(
    float* grad_relu,
    const float* grad_output,
    const float* x,
    const float* weights,
    const float* biases,
    int batch_size,
    int N1,
    int N2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * N2) return;

    int i = tid / N2;
    int j = tid % N2;

    float y_linear = 0.0f;
    for (int k = 0; k < N1; ++k) {
        float x_val = x[i * N1 + k];
        float w_val = weights[j * N1 + k];
        y_linear += x_val * w_val;
    }
    y_linear += biases[j];

    int idx = i * N2 + j;
    float grad_out_val = grad_output[idx];

    float mask = (y_linear > 0.0f) ? 1.0f : 0.0f;
    grad_relu[idx] = grad_out_val * mask;
}

// CUDA kernel to compute grad_input
__global__ void compute_grad_input(
    float* grad_input,
    const float* grad_relu,
    const float* weights,
    int batch_size,
    int N1,
    int N2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * N1) return;

    int i = tid / N1;
    int k = tid % N1;

    float sum = 0.0f;
    for (int j = 0; j < N2; ++j) {
        int idx = i * N2 + j;
        sum += grad_relu[idx] * weights[j * N1 + k];
    }

    grad_input[i * N1 + k] = sum;
}

// CUDA kernel to compute grad_weights and grad_biases
__global__ void compute_grad_weights_and_biases(
    float* grad_weights,
    float* grad_biases,
    const float* grad_relu,
    const float* x,
    int batch_size,
    int N1,
    int N2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = N2 * N1 + N2;

    if (tid >= total_work) return;

    if (tid < N2 * N1) {
        int j = tid / N1;
        int k = tid % N1;

        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            int idx = i * N2 + j;
            sum += grad_relu[idx] * x[i * N1 + k];
        }

        grad_weights[j * N1 + k] = sum;
    } else {
        int j = tid - N2 * N1;

        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            int idx = i * N2 + j;
            sum += grad_relu[idx];
        }

        grad_biases[j] = sum;
    }
}

// C++ function to be exposed to Python
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor biases
) {
    int batch_size = x.size(0);
    int N1 = x.size(1);
    int N2 = weights.size(0);

    // Create gradient output tensors
    auto grad_input = torch::empty({batch_size, N1}, x.options());
    auto grad_weights = torch::empty({N2, N1}, weights.options());
    auto grad_biases = torch::empty({N2}, biases.options());

    // Temporary storage for grad_relu
    auto grad_relu = torch::empty({batch_size, N2}, x.options());

    // Kernel launch parameters
    int threads_per_block = 256;

    // Compute grad_relu
    int num_threads_relu = batch_size * N2;
    int blocks_relu = (num_threads_relu + threads_per_block - 1) / threads_per_block;

    compute_grad_relu<<<blocks_relu, threads_per_block>>>(
        grad_relu.data<float>(),
        grad_output.data<float>(),
        x.data<float>(),
        weights.data<float>(),
        biases.data<float>(),
        batch_size,
        N1,
        N2
    );

    // Compute grad_input
    int num_threads_input = batch_size * N1;
    int blocks_input = (num_threads_input + threads_per_block - 1) / threads_per_block;

    compute_grad_input<<<blocks_input, threads_per_block>>>(
        grad_input.data<float>(),
        grad_relu.data<float>(),
        weights.data<float>(),
        batch_size,
        N1,
        N2
    );

    // Compute grad_weights and grad_biases
    int total_work_weights_biases = N2 * N1 + N2;
    int blocks_weights_biases = (total_work_weights_biases + threads_per_block - 1) / threads_per_block;

    compute_grad_weights_and_biases<<<blocks_weights_biases, threads_per_block>>>(
        grad_weights.data<float>(),
        grad_biases.data<float>(),
        grad_relu.data<float>(),
        x.data<float>(),
        batch_size,
        N1,
        N2
    );

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU backward using CUDA");
}