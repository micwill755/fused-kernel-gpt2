// Step 1: Simple CUDA Kernel - Element-wise Addition
// This is your first CUDA kernel to understand the basics

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel function - runs on GPU
__global__ void simple_add_kernel(
    const float* input_a,    // Input tensor A
    const float* input_b,    // Input tensor B  
    float* output,           // Output tensor
    int num_elements         // Total number of elements
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check - make sure we don't go out of bounds
    if (idx < num_elements) {
        // Simple element-wise addition
        output[idx] = input_a[idx] + input_b[idx];
    }
}

// Host function - runs on CPU, launches GPU kernel
torch::Tensor simple_add_cuda(torch::Tensor input_a, torch::Tensor input_b) {
    // Get tensor dimensions
    auto num_elements = input_a.numel();
    
    // Create output tensor with same shape as input
    auto output = torch::zeros_like(input_a);
    
    // Define CUDA launch parameters
    const int threads_per_block = 256;  // Common choice: 256 threads per block
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch the CUDA kernel
    simple_add_kernel<<<blocks, threads_per_block>>>(
        input_a.data_ptr<float>(),   // Get raw pointer to tensor data
        input_b.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    // Check for CUDA errors (important for debugging!)
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    return output;
}

// Python binding - makes the function callable from Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simple_add", &simple_add_cuda, "Simple element-wise addition (CUDA)");
}
