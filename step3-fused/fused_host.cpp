#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>

// Forward declarations of CUDA kernels
extern "C" {
    void launch_fused_attention_layernorm(
        const float* input,
        const float* Q_weight,
        const float* K_weight,
        const float* V_weight,
        const float* ln_weight,
        const float* ln_bias,
        float* output,
        float* temp_scores,
        int batch_size,
        int seq_len,
        int hidden_size,
        int num_heads,
        float eps,
        cudaStream_t stream
    );
    
    void launch_optimized_fused_attention_layernorm(
        const float* input,
        const float* qkv_weight,
        const float* ln_weight,
        const float* ln_bias,
        float* output,
        int batch_size,
        int seq_len,
        int hidden_size,
        int num_heads,
        float eps,
        cudaStream_t stream
    );
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

torch::Tensor fused_attention_layernorm_forward(
    torch::Tensor input,
    torch::Tensor Q_weight,
    torch::Tensor K_weight,
    torch::Tensor V_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int num_heads,
    float eps = 1e-5f,
    bool use_optimized = false
) {
    // Input validation
    check_tensor(input, "input");
    check_tensor(Q_weight, "Q_weight");
    check_tensor(K_weight, "K_weight");
    check_tensor(V_weight, "V_weight");
    check_tensor(ln_weight, "ln_weight");
    check_tensor(ln_bias, "ln_bias");
    
    // Get dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int seq_len = input_sizes[1];
    int hidden_size = input_sizes[2];
    
    TORCH_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");
    TORCH_CHECK(Q_weight.sizes() == torch::IntArrayRef({hidden_size, hidden_size}), "Q_weight shape mismatch");
    TORCH_CHECK(K_weight.sizes() == torch::IntArrayRef({hidden_size, hidden_size}), "K_weight shape mismatch");
    TORCH_CHECK(V_weight.sizes() == torch::IntArrayRef({hidden_size, hidden_size}), "V_weight shape mismatch");
    TORCH_CHECK(ln_weight.sizes() == torch::IntArrayRef({hidden_size}), "ln_weight shape mismatch");
    TORCH_CHECK(ln_bias.sizes() == torch::IntArrayRef({hidden_size}), "ln_bias shape mismatch");
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Create temporary tensor for attention scores
    auto temp_scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, 
                                   torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    if (use_optimized) {
        // For optimized version, we need combined QKV weights
        auto qkv_weight = torch::stack({Q_weight, K_weight, V_weight}, 0);
        
        launch_optimized_fused_attention_layernorm(
            input.data_ptr<float>(),
            qkv_weight.data_ptr<float>(),
            ln_weight.data_ptr<float>(),
            ln_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, seq_len, hidden_size, num_heads, eps,
            stream
        );
    } else {
        launch_fused_attention_layernorm(
            input.data_ptr<float>(),
            Q_weight.data_ptr<float>(),
            K_weight.data_ptr<float>(),
            V_weight.data_ptr<float>(),
            ln_weight.data_ptr<float>(),
            ln_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            temp_scores.data_ptr<float>(),
            batch_size, seq_len, hidden_size, num_heads, eps,
            stream
        );
    }
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

// Reference implementation using separate operations
torch::Tensor reference_attention_layernorm(
    torch::Tensor input,
    torch::Tensor Q_weight,
    torch::Tensor K_weight,
    torch::Tensor V_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int num_heads,
    float eps = 1e-5f
) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / std::sqrt(head_dim);
    
    // Linear projections
    auto Q = torch::matmul(input, Q_weight.t());
    auto K = torch::matmul(input, K_weight.t());
    auto V = torch::matmul(input, V_weight.t());
    
    // Reshape for multi-head attention
    Q = Q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    K = K.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    V = V.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    
    // Compute attention
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;
    auto attn_weights = torch::softmax(scores, -1);
    auto attn_output = torch::matmul(attn_weights, V);
    
    // Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_size});
    
    // Add residual connection
    auto residual_output = attn_output + input;
    
    // Layer normalization
    auto output = torch::layer_norm(residual_output, {hidden_size}, ln_weight, ln_bias, eps);
    
    return output;
}

// Benchmark function
std::vector<float> benchmark_fused_kernel(
    torch::Tensor input,
    torch::Tensor Q_weight,
    torch::Tensor K_weight,
    torch::Tensor V_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int num_heads,
    float eps = 1e-5f,
    int num_iterations = 100
) {
    // Warm up
    for (int i = 0; i < 10; i++) {
        auto result = fused_attention_layernorm_forward(
            input, Q_weight, K_weight, V_weight, ln_weight, ln_bias, num_heads, eps, false);
        torch::cuda::synchronize();
    }
    
    // Benchmark basic fused kernel
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = fused_attention_layernorm_forward(
            input, Q_weight, K_weight, V_weight, ln_weight, ln_bias, num_heads, eps, false);
    }
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float fused_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark optimized fused kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = fused_attention_layernorm_forward(
            input, Q_weight, K_weight, V_weight, ln_weight, ln_bias, num_heads, eps, true);
    }
    torch::cuda::synchronize();
    end = std::chrono::high_resolution_clock::now();
    float optimized_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark reference implementation
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto result = reference_attention_layernorm(
            input, Q_weight, K_weight, V_weight, ln_weight, ln_bias, num_heads, eps);
    }
    torch::cuda::synchronize();
    end = std::chrono::high_resolution_clock::now();
    float reference_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    return {fused_time, optimized_time, reference_time};
}

// Memory bandwidth analysis
std::vector<float> analyze_memory_bandwidth(
    torch::Tensor input,
    torch::Tensor Q_weight,
    torch::Tensor K_weight,
    torch::Tensor V_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    int num_heads,
    float eps = 1e-5f
) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    
    // Calculate theoretical memory requirements (in bytes)
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t weight_size = 3 * hidden_size * hidden_size * sizeof(float); // Q, K, V weights
    size_t ln_param_size = 2 * hidden_size * sizeof(float); // ln_weight, ln_bias
    size_t output_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t temp_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    
    // Fused kernel: input + weights + ln_params + output (no intermediate storage)
    float fused_memory_gb = (input_size + weight_size + ln_param_size + output_size) / (1024.0f * 1024.0f * 1024.0f);
    
    // Reference: input + weights + ln_params + output + intermediate tensors
    float reference_memory_gb = (input_size + weight_size + ln_param_size + output_size + 
                                3 * input_size + temp_size) / (1024.0f * 1024.0f * 1024.0f);
    
    // Memory bandwidth savings
    float savings_percent = (1.0f - fused_memory_gb / reference_memory_gb) * 100.0f;
    
    return {fused_memory_gb, reference_memory_gb, savings_percent};
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention_layernorm", &fused_attention_layernorm_forward, 
          "Fused Attention + LayerNorm Forward",
          py::arg("input"), py::arg("Q_weight"), py::arg("K_weight"), py::arg("V_weight"),
          py::arg("ln_weight"), py::arg("ln_bias"), py::arg("num_heads"), 
          py::arg("eps") = 1e-5f, py::arg("use_optimized") = false);
    
    m.def("reference_attention_layernorm", &reference_attention_layernorm,
          "Reference Attention + LayerNorm Implementation",
          py::arg("input"), py::arg("Q_weight"), py::arg("K_weight"), py::arg("V_weight"),
          py::arg("ln_weight"), py::arg("ln_bias"), py::arg("num_heads"), py::arg("eps") = 1e-5f);
    
    m.def("benchmark_fused", &benchmark_fused_kernel,
          "Benchmark Fused Kernel vs Reference",
          py::arg("input"), py::arg("Q_weight"), py::arg("K_weight"), py::arg("V_weight"),
          py::arg("ln_weight"), py::arg("ln_bias"), py::arg("num_heads"), 
          py::arg("eps") = 1e-5f, py::arg("num_iterations") = 100);
    
    m.def("analyze_memory", &analyze_memory_bandwidth,
          "Analyze Memory Bandwidth Usage",
          py::arg("input"), py::arg("Q_weight"), py::arg("K_weight"), py::arg("V_weight"),
          py::arg("ln_weight"), py::arg("ln_bias"), py::arg("num_heads"), py::arg("eps") = 1e-5f);
}
