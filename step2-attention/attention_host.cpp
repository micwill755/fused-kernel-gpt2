#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <random>

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

// Pure C++ attention forward function
void cuda_attention_forward(
    const float* Q, const float* K, const float* V,
    float* output, float* temp_scores,
    int batch_size, int seq_len, int hidden_size, int num_heads,
    bool use_optimized = false
) {
    // Validate inputs
    assert(Q != nullptr && K != nullptr && V != nullptr && output != nullptr);
    assert(batch_size > 0 && seq_len > 0 && hidden_size > 0 && num_heads > 0);
    assert(hidden_size % num_heads == 0);
    
    // Get CUDA stream (use default stream)
    cudaStream_t stream = 0;
    
    // Launch appropriate kernel
    if (use_optimized) {
        launch_optimized_attention_kernel(
            Q, K, V, output, temp_scores,
            batch_size, seq_len, hidden_size, num_heads,
            stream
        );
    } else {
        launch_attention_kernel(
            Q, K, V, output, temp_scores,
            batch_size, seq_len, hidden_size, num_heads,
            stream
        );
    }
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Benchmark function - pure C++
std::vector<float> benchmark_attention(
    const float* Q, const float* K, const float* V,
    float* output, float* temp_scores,
    int batch_size, int seq_len, int hidden_size, int num_heads,
    int num_iterations = 100
) {
    // Warm up
    for (int i = 0; i < 10; i++) {
        cuda_attention_forward(Q, K, V, output, temp_scores,
                              batch_size, seq_len, hidden_size, num_heads, false);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark basic kernel
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        cuda_attention_forward(Q, K, V, output, temp_scores,
                              batch_size, seq_len, hidden_size, num_heads, false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    float basic_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark optimized kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        cuda_attention_forward(Q, K, V, output, temp_scores,
                              batch_size, seq_len, hidden_size, num_heads, true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    float optimized_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    return {basic_time, optimized_time};
}

// Utility function to initialize test data
void initialize_random_data(float* data, int size, float scale = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

// Utility function to compare results
bool compare_results(const float* a, const float* b, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}
