#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_dy_relu(
    const float* x,
    const float* weights,
    const float* biases,
    const float* grad_output,
    float* dy_relu,
    int B,
    int I,
    int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * O) return;

    int i = idx / O;
    int j = idx % O;

    float y_linear = 0.0f;
    for (int k = 0; k < I; ++k) {
        y_linear += x[i * I + k] * weights[j * I + k];
    }
    y_linear += biases[j];

    float mask = (y_linear > 0.0f) ? 1.0f : 0.0f;
    dy_relu[idx] = grad_output[idx] * mask;
}

__global__ void compute_dx(
    const float* dy_relu,
    const float* weights,
    float* dx,
    int B,
    int I,
    int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * I) return;

    int i = idx / I;
    int k = idx % I;

    float sum = 0.0f;
    for (int j = 0; j < O; ++j) {
        int dy_idx = i * O + j;
        int w_idx = j * I + k;
        sum += dy_relu[dy_idx] * weights[w_idx];
    }
    dx[idx] = sum;
}

__global__ void compute_dw(
    const float* dy_relu,
    const float* x,
    float* dw,
    int B,
    int I,
    int O
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= O * I) return;

    int j = idx / I;
    int k = idx % I;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        int dy_idx = i * O + j;
        int x_idx = i * I + k;
        sum += dy_relu[dy_idx] * x[x_idx];
    }
    dw[idx] = sum;
}

__global__ void compute_db(
    const float* dy_relu,
    float* db,
    int B,
    int O
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= O) return;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        int idx = i * O + j;
        sum += dy_relu[idx];
    }
    db[j] = sum;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor biases
) {
    int B = x.size(0);
    int I = x.size(1);
    int O = weights.size(0);

    auto options = x.options();
    torch::Tensor grad_input = torch::empty({B, I}, options);
    torch::Tensor grad_weights = torch::empty({O, I}, options);
    torch::Tensor grad_biases = torch::empty({O}, options);
    torch::Tensor dy_relu = torch::empty({B, O}, options);

    dim3 threads_per_block(256);
    int total_dy_relu = B * O;
    int num_blocks_dy_relu = (total_dy_relu + threads_per_block.x - 1) / threads_per_block.x;
    compute_dy_relu<<<num_blocks_dy_relu, threads_per_block>>>(
        x.data_ptr<float>(),
        weights.data_ptr<float>(),
        biases.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        dy_relu.data_ptr<float>(),
        B, I, O
    );

    int total_dx = B * I;
    int num_blocks_dx = (total_dx + threads_per_block.x - 1) / threads_per_block.x;
    compute_dx<<<num_blocks_dx, threads_per_block>>>(
        dy_relu.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        B, I, O
    );

    int total_dw = O * I;
    int num_blocks_dw = (total_dw + threads_per_block.x - 1) / threads_per_block.x;
    compute_dw<<<num_blocks_dw, threads_per_block>>>(
        dy_relu.data_ptr<float>(),
        x.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        B, I, O
    );

    int total_db = O;
    int num_blocks_db = (total_db + threads_per_block.x - 1) / threads_per_block.x;
    compute_db<<<num_blocks_db, threads_per_block>>>(
        dy_relu.data_ptr<float>(),
        grad_biases.data_ptr<float>(),
        B, O
    );

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}