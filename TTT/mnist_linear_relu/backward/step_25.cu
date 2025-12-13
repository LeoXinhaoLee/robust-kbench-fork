#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute g (grad_output multiplied by ReLU mask)
__global__ void compute_g_kernel(
    const float* x,
    const float* weights,
    const float* biases,
    const float* grad_output,
    float* g,
    int B, int N, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;

    int i = idx / N;
    int j = idx % N;

    float y = 0.0f;
    for (int k = 0; k < K; ++k) {
        y += x[i * K + k] * weights[j * K + k];
    }
    y += biases[j];

    float mask = (y > 0.0f) ? 1.0f : 0.0f;
    float go_val = grad_output[i * N + j];
    g[idx] = go_val * mask;
}

// Kernel for grad_input
__global__ void grad_input_kernel(
    const float* g,
    const float* weights,
    float* grad_input,
    int B, int N, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * K) return;

    int i = idx / K;
    int k = idx % K;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        int g_idx = i * N + j;
        sum += g[g_idx] * weights[j * K + k];
    }
    grad_input[idx] = sum;
}

// Kernel for grad_weights
__global__ void grad_weights_kernel(
    const float* x,
    const float* g,
    float* grad_weights,
    int B, int N, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int j = idx / K;
    int k = idx % K;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        int x_idx = i * K + k;
        int g_idx = i * N + j;
        sum += x[x_idx] * g[g_idx];
    }
    grad_weights[idx] = sum;
}

// Kernel for grad_biases
__global__ void grad_biases_kernel(
    const float* g,
    float* grad_biases,
    int B, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int j = idx;

    float sum = 0.0f;
    for (int i = 0; i < B; ++i) {
        int g_idx = i * N + j;
        sum += g[g_idx];
    }
    grad_biases[j] = sum;
}

// C++ function to be exposed to Python
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor weights, 
    torch::Tensor biases) {
    
    int B = x.size(0); // batch size
    int K = x.size(1); // num_input_features
    int N = weights.size(0); // num_output_features

    auto options = torch::TensorOptions()
        .device(x.device())
        .dtype(x.dtype());

    auto grad_input_tensor = torch::empty({B, K}, options);
    auto grad_weights_tensor = torch::empty({N, K}, options);
    auto grad_biases_tensor = torch::empty({N}, options);
    auto g_tensor = torch::empty({B, N}, options);

    float* g = g_tensor.data_ptr<float>();
    const float* x_data = x.data_ptr<float>();
    const float* weights_data = weights.data_ptr<float>();
    const float* biases_data = biases.data_ptr<float>();
    const float* grad_output_data = grad_output.data_ptr<float>();
    float* grad_input_data = grad_input_tensor.data_ptr<float>();
    float* grad_weights_data = grad_weights_tensor.data_ptr<float>();
    float* grad_biases_data = grad_biases_tensor.data_ptr<float>();

    int threads_per_block = 256;

    // Launch compute_g kernel
    int total_g = B * N;
    int blocks_g = (total_g + threads_per_block - 1) / threads_per_block;
    compute_g_kernel<<<blocks_g, threads_per_block>>>(
        x_data, weights_data, biases_data, grad_output_data, g, B, N, K);

    // Launch grad_input kernel
    int total_grad_input = B * K;
    int blocks_grad_input = (total_grad_input + threads_per_block - 1) / threads_per_block;
    grad_input_kernel<<<blocks_grad_input, threads_per_block>>>(
        g, weights_data, grad_input_data, B, N, K);

    // Launch grad_weights kernel
    int total_grad_weights = N * K;
    int blocks_grad_weights = (total_grad_weights + threads_per_block - 1) / threads_per_block;
    grad_weights_kernel<<<blocks_grad_weights, threads_per_block>>>(
        x_data, g, grad_weights_data, B, N, K);

    // Launch grad_biases kernel
    int total_grad_biases = N;
    int blocks_grad_biases = (total_grad_biases + threads_per_block - 1) / threads_per_block;
    grad_biases_kernel<<<blocks_grad_biases, threads_per_block>>>(
        g, grad_biases_data, B, N);

    return {grad_input_tensor, grad_weights_tensor, grad_biases_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}