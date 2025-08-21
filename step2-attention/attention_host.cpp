#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Forward declarations of CUDA kernels
extern "C" {
    void launch_attention_kernel(const float* Q, const float* K, const float* V,
                                float* output, float* temp_scores,
                                int batch_size, int seq_len, int hidden_size,
                                int num_heads, cudaStream_t stream);
    
    void launch_optimized_attention_kernel(const float* Q, const float* K, const float* V,
                                          float* output, float* temp_scores,
                                          int batch_size, int seq_len, int hidden_size,
                                          int num_heads, cudaStream_t stream);
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Check tensor properties
void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name + " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name + " must be contiguous");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name + " must be float32");
}

torch::Tensor cuda_attention_forward(
    torch::Tensor Q,
    torch::Tensor K, 
    torch::Tensor V,
    int num_heads,
    bool use_optimized = false
) {
    // Input validation
    check_tensor(Q, "Q");
    check_tensor(K, "K");
    check_tensor(V, "V");
    
    // Get dimensions
    auto Q_sizes = Q.sizes();
    int batch_size = Q_sizes[0];
    int seq_len = Q_sizes[1];
    int hidden_size = Q_sizes[2];
    
    TORCH_CHECK(Q_sizes == K.sizes(), "Q and K must have same shape");
    TORCH_CHECK(Q_sizes == V.sizes(), "Q and V must have same shape");
    TORCH_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");
    
    // Create output tensor
    auto output = torch::zeros_like(Q);
    
    // Create temporary tensor for attention scores
    auto temp_scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    if (use_optimized) {
        launch_optimized_attention_kernel(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            output.data_ptr<float>(),
            temp_scores.data_ptr<float>(),
            batch_size, seq_len, hidden_size, num_heads,
            stream
        );
    } else {
        launch_attention_kernel(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            output.data_ptr<float>(),
            temp_scores.data_ptr<float>(),
            batch_size, seq_len, hidden_size, num_heads,
            stream
        );
    }
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// Benchmark function
std::vector<float> benchmark_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int num_heads,
    int num_iterations = 100
) {
    check_tensor(Q, "Q");
    check_tensor(K, "K");
    check_tensor(V, "V");
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        auto result = cuda_attention_forward(Q, K, V, num_heads, false);
        torch::cuda::synchronize();
    }
    
    // Benchmark basic kernel
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = cuda_attention_forward(Q, K, V, num_heads, false);
    }
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float basic_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark optimized kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = cuda_attention_forward(Q, K, V, num_heads, true);
    }
    torch::cuda::synchronize();
    end = std::chrono::high_resolution_clock::now();
    float optimized_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark PyTorch reference
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        int head_dim = Q.size(2) / num_heads;
        float scale = 1.0f / std::sqrt(head_dim);
        
        // Reshape for multi-head attention
        auto Q_heads = Q.view({Q.size(0), Q.size(1), num_heads, head_dim}).transpose(1, 2);
        auto K_heads = K.view({K.size(0), K.size(1), num_heads, head_dim}).transpose(1, 2);
        auto V_heads = V.view({V.size(0), V.size(1), num_heads, head_dim}).transpose(1, 2);
        
        // Compute attention
        auto scores = torch::matmul(Q_heads, K_heads.transpose(-2, -1)) * scale;
        auto attn_weights = torch::softmax(scores, -1);
        auto result = torch::matmul(attn_weights, V_heads);
        
        // Reshape back
        result = result.transpose(1, 2).contiguous().view({Q.size(0), Q.size(1), Q.size(2)});
    }
    torch::cuda::synchronize();
    end = std::chrono::high_resolution_clock::now();
    float pytorch_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    return {basic_time, optimized_time, pytorch_time};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &cuda_attention_forward, "CUDA Attention Forward",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("num_heads"), py::arg("use_optimized") = false);
    
    m.def("benchmark_attention", &benchmark_attention, "Benchmark Attention Kernels",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("num_heads"), py::arg("num_iterations") = 100);
}
