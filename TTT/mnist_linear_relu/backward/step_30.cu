#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute grad_relu
__global__ void compute_grad_relu(const float* x, const float* weights, const float* biases,
                                  const float* grad_output, float* grad_relu,
                                  int B, int out, int in) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= B * out) return;

    int i = idx / out;
    int j = idx % out;

    float sum = 0.0f;
    for (int k = 0; k < in; ++k) {
        sum += x[i * in + k] * weights[j * in + k];
    }
    sum += biases[j];

    float mask = (sum > 0.0f) ? 1.0f : 0.0f;
    grad_relu[idx] = grad_output[i * out + j] * mask;
}

// Kernel to compute grad_input
__global__ void compute_grad_input(const float* grad_relu, const float* weights,
                                   float* grad_input, int B, int out, int in) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= B * in) return;

    int i = idx / in;
    int k_prime = idx % in;

    float sum = 0.0f;
    for (int j = 0; j < out; ++j) {
        float gr = grad_relu[i * out + j];
        float w = weights[j * in + k_prime];
        sum += gr * w;
    }
    grad_input[idx] = sum;
}

// Kernel to compute grad_weights
__global__ void compute_grad_weights(const float* grad_relu, const float* x,
                                     float* grad_weights, int B, int out, int in) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= out * in) return;

    int j = idx / in;
    int k_prime = idx % in;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        float gr = grad_relu[i * out + j];
        float x_val = x[i * in + k_prime];
        sum += gr * x_val;
    }
    grad_weights[idx] = sum;
}

// Kernel to compute grad_biases
__global__ void compute_grad_biases(const float* grad_relu, float* grad_biases,
                                    int B, int out) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= out) return;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        sum += grad_relu[i * out + j];
    }
    grad_biases[j] = sum;
}

// C++ function to be exposed to Python
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights, torch::Tensor biases) {
    int B = x.size(0);
    int in = x.size(1);
    int out = weights.size(0);

    auto options = x.options(); // same device and dtype as x

    auto grad_relu = torch::empty({B, out}, options);
    auto grad_input = torch::empty({B, in}, options);
    auto grad_weights = torch::empty({out, in}, options);
    auto grad_biases = torch::empty({out}, options);

    // Launch compute_grad_relu
    int threads_per_block = 256;
    int num_threads_relu = B * out;
    int blocks_relu = (num_threads_relu + threads_per_block - 1) / threads_per_block;
    compute_grad_relu<<<blocks_relu, threads_per_block>>>(
        x.data_ptr<float>(), weights.data_ptr<float>(), biases.data_ptr<float>(),
        grad_output.data_ptr<float>(), grad_relu.data_ptr<float>(), B, out, in);

    // Launch compute_grad_input
    int num_threads_input = B * in;
    int blocks_input = (num_threads_input + threads_per_block - 1) / threads_per_block;
    compute_grad_input<<<blocks_input, threads_per_block>>>(
        grad_relu.data_ptr<float>(), weights.data_ptr<float>(), grad_input.data_ptr<float>(), B, out, in);

    // Launch compute_grad_weights
    int num_threads_weights = out * in;
    int blocks_weights = (num_threads_weights + threads_per_block - 1) / threads_per_block;
    compute_grad_weights<<<blocks_weights, threads_per_block>>>(
        grad_relu.data_ptr<float>(), x.data_ptr<float>(), grad_weights.data_ptr<float>(), B, out, in);

    // Launch compute_grad_biases
    int num_threads_biases = out;
    int blocks_biases = (num_threads_biases + threads_per_block - 1) / threads_per_block;
    compute_grad_biases<<<blocks_biases, threads_per_block>>>(
        grad_relu.data_ptr<float>(), grad_biases.data_ptr<float>(), B, out);

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Backward function for linear ReLU");
}