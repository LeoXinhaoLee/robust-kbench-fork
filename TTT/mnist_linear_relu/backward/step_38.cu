#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_grad_linear(
    const float* grad_output,
    const float* x,
    const float* weights,
    const float* biases,
    float* grad_linear,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int b = idx / out_features;
    int o = idx % out_features;

    float z = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        z += x[b * in_features + i] * weights[o * in_features + i];
    }
    z += biases[o];

    float mask = (z > 0.0f) ? 1.0f : 0.0f;

    grad_linear[idx] = grad_output[b * out_features + o] * mask;
}

__global__ void compute_grad_input(
    const float* grad_linear,
    const float* weights,
    float* grad_input,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_features) return;

    int b = idx / in_features;
    int i = idx % in_features;

    float sum = 0.0f;
    for (int o = 0; o < out_features; ++o) {
        int gl_idx = b * out_features + o;
        int w_idx = o * in_features + i;
        sum += grad_linear[gl_idx] * weights[w_idx];
    }
    grad_input[idx] = sum;
}

__global__ void compute_grad_weights_and_biases(
    const float* grad_linear,
    const float* x,
    float* grad_weights,
    float* grad_biases,
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_features * in_features) return;

    int o = idx / in_features;
    int i = idx % in_features;

    float sum_weights = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        int gl_idx = b * out_features + o;
        sum_weights += grad_linear[gl_idx] * x[b * in_features + i];
    }
    grad_weights[idx] = sum_weights;

    if (i == 0) {
        float sum_biases = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            int gl_idx = b * out_features + o;
            sum_biases += grad_linear[gl_idx];
        }
        grad_biases[o] = sum_biases;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights, torch::Tensor biases) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weights.size(0);

    auto grad_linear = torch::empty({batch_size, out_features}, x.options());
    auto grad_input = torch::empty({batch_size, in_features}, x.options());
    auto grad_weights = torch::empty({out_features, in_features}, weights.options());
    auto grad_biases = torch::empty({out_features}, biases.options());

    const float* grad_output_data = grad_output.data<float>();
    const float* x_data = x.data<float>();
    const float* weights_data = weights.data<float>();
    const float* biases_data = biases.data<float>();

    float* grad_linear_data = grad_linear.data<float>();
    float* grad_input_data = grad_input.data<float>();
    float* grad_weights_data = grad_weights.data<float>();
    float* grad_biases_data = grad_biases.data<float>();

    int num_threads = 256;
    int num_blocks;

    // Compute grad_linear
    num_blocks = (batch_size * out_features + num_threads - 1) / num_threads;
    compute_grad_linear<<<num_blocks, num_threads>>>(
        grad_output_data, x_data, weights_data, biases_data, grad_linear_data,
        batch_size, in_features, out_features
    );

    // Compute grad_input
    num_blocks = (batch_size * in_features + num_threads - 1) / num_threads;
    compute_grad_input<<<num_blocks, num_threads>>>(
        grad_linear_data, weights_data, grad_input_data,
        batch_size, in_features, out_features
    );

    // Compute grad_weights and grad_biases
    num_blocks = (out_features * in_features + num_threads - 1) / num_threads;
    compute_grad_weights_and_biases<<<num_blocks, num_threads>>>(
        grad_linear_data, x_data, grad_weights_data, grad_biases_data,
        batch_size, in_features, out_features
    );

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}