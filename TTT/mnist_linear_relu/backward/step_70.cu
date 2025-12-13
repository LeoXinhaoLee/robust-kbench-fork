#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to compute dz
__global__ void compute_dz_kernel(
    const float* grad_output, 
    const float* x, 
    const float* weights, 
    const float* biases, 
    float* dz,
    int batch_size, 
    int num_inputs, 
    int num_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_outputs) return;

    int batch = idx / num_outputs;
    int output = idx % num_outputs;

    float sum_x_w = 0.0f;
    for (int input = 0; input < num_inputs; input++) {
        int x_idx = batch * num_inputs + input;
        int w_idx = output * num_inputs + input;
        sum_x_w += x[x_idx] * weights[w_idx];
    }

    float z = sum_x_w + biases[output];
    float mask = (z > 0.0f) ? 1.0f : 0.0f;

    int grad_output_idx = batch * num_outputs + output;
    float dz_value = grad_output[grad_output_idx] * mask;

    dz[idx] = dz_value;
}

// CUDA kernel to compute grad_input
__global__ void compute_grad_input_kernel(
    const float* dz, 
    const float* weights, 
    float* grad_input,
    int batch_size, 
    int num_inputs, 
    int num_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_inputs) return;

    int batch = idx / num_inputs;
    int input = idx % num_inputs;

    float sum = 0.0f;
    for (int output = 0; output < num_outputs; output++) {
        int dz_idx = batch * num_outputs + output;
        int w_idx = output * num_inputs + input;
        sum += dz[dz_idx] * weights[w_idx];
    }

    grad_input[idx] = sum;
}

// CUDA kernel to compute grad_weights
__global__ void compute_grad_weights_kernel(
    const float* dz, 
    const float* x, 
    float* grad_weights,
    int batch_size, 
    int num_inputs, 
    int num_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs * num_inputs) return;

    int output = idx / num_inputs;
    int input = idx % num_inputs;

    float sum = 0.0f;
    for (int batch = 0; batch < batch_size; batch++) {
        int dz_idx = batch * num_outputs + output;
        int x_idx = batch * num_inputs + input;
        sum += dz[dz_idx] * x[x_idx];
    }

    grad_weights[idx] = sum;
}

// CUDA kernel to compute grad_biases
__global__ void compute_grad_biases_kernel(
    const float* dz, 
    float* grad_biases,
    int batch_size, 
    int num_outputs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_outputs) return;

    int output = idx;
    float sum = 0.0f;
    for (int batch = 0; batch < batch_size; batch++) {
        int dz_idx = batch * num_outputs + output;
        sum += dz[dz_idx];
    }

    grad_biases[output] = sum;
}

// C++ function to be exposed to Python
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor weights, 
    torch::Tensor biases
) {
    int batch_size = x.size(0);
    int num_inputs = x.size(1);
    int num_outputs = weights.size(0);

    auto options = x.options();
    auto grad_input = torch::empty({batch_size, num_inputs}, options);
    auto grad_weights = torch::empty({num_outputs, num_inputs}, options);
    auto grad_biases = torch::empty({num_outputs}, options);

    auto dz = torch::empty({batch_size * num_outputs}, options);

    // Launch compute_dz_kernel
    int threads_per_block = 256;
    int total_threads_dz = batch_size * num_outputs;
    int blocks_dz = (total_threads_dz + threads_per_block - 1) / threads_per_block;
    compute_dz_kernel<<<blocks_dz, threads_per_block>>>(
        grad_output.data<float>(),
        x.data<float>(),
        weights.data<float>(),
        biases.data<float>(),
        dz.data<float>(),
        batch_size, num_inputs, num_outputs
    );

    // Launch compute_grad_input_kernel
    int total_threads_grad_input = batch_size * num_inputs;
    int blocks_grad_input = (total_threads_grad_input + threads_per_block - 1) / threads_per_block;
    compute_grad_input_kernel<<<blocks_grad_input, threads_per_block>>>(
        dz.data<float>(),
        weights.data<float>(),
        grad_input.data<float>(),
        batch_size, num_inputs, num_outputs
    );

    // Launch compute_grad_weights_kernel
    int total_threads_grad_weights = num_outputs * num_inputs;
    int blocks_grad_weights = (total_threads_grad_weights + threads_per_block - 1) / threads_per_block;
    compute_grad_weights_kernel<<<blocks_grad_weights, threads_per_block>>>(
        dz.data<float>(),
        x.data<float>(),
        grad_weights.data<float>(),
        batch_size, num_inputs, num_outputs
    );

    // Launch compute_grad_biases_kernel
    int total_threads_grad_biases = num_outputs;
    int blocks_grad_biases = (total_threads_grad_biases + threads_per_block - 1) / threads_per_block;
    compute_grad_biases_kernel<<<blocks_grad_biases, threads_per_block>>>(
        dz.data<float>(),
        grad_biases.data<float>(),
        batch_size, num_outputs
    );

    return {grad_input, grad_weights, grad_biases};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Linear ReLU Backward");
}