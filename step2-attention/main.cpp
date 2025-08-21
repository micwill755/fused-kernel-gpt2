#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel declarations
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
            exit(1); \
        } \
    } while(0)

class AttentionTester {
private:
    int batch_size_, seq_len_, hidden_size_, num_heads_;
    float* h_Q_, *h_K_, *h_V_, *h_output_, *h_reference_;
    float* d_Q_, *d_K_, *d_V_, *d_output_, *d_temp_scores_;
    
public:
    AttentionTester(int batch_size, int seq_len, int hidden_size, int num_heads)
        : batch_size_(batch_size), seq_len_(seq_len), hidden_size_(hidden_size), num_heads_(num_heads) {
        
        size_t qkv_size = batch_size * seq_len * hidden_size * sizeof(float);
        size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
        
        // Allocate host memory
        h_Q_ = new float[batch_size * seq_len * hidden_size];
        h_K_ = new float[batch_size * seq_len * hidden_size];
        h_V_ = new float[batch_size * seq_len * hidden_size];
        h_output_ = new float[batch_size * seq_len * hidden_size];
        h_reference_ = new float[batch_size * seq_len * hidden_size];
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_Q_, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_K_, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_V_, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_output_, qkv_size));
        CUDA_CHECK(cudaMalloc(&d_temp_scores_, scores_size));
        
        // Initialize random data
        initializeRandomData();
    }
    
    ~AttentionTester() {
        delete[] h_Q_;
        delete[] h_K_;
        delete[] h_V_;
        delete[] h_output_;
        delete[] h_reference_;
        
        cudaFree(d_Q_);
        cudaFree(d_K_);
        cudaFree(d_V_);
        cudaFree(d_output_);
        cudaFree(d_temp_scores_);
    }
    
    void initializeRandomData() {
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        int total_elements = batch_size_ * seq_len_ * hidden_size_;
        
        for (int i = 0; i < total_elements; i++) {
            h_Q_[i] = dist(gen);
            h_K_[i] = dist(gen);
            h_V_[i] = dist(gen);
        }
        
        // Copy to device
        size_t qkv_size = total_elements * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_Q_, h_Q_, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K_, h_K_, qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_, h_V_, qkv_size, cudaMemcpyHostToDevice));
    }
    
    void computeReferenceAttention() {
        // CPU reference implementation
        int head_dim = hidden_size_ / num_heads_;
        float scale = 1.0f / std::sqrt(head_dim);
        
        for (int b = 0; b < batch_size_; b++) {
            for (int h = 0; h < num_heads_; h++) {
                for (int i = 0; i < seq_len_; i++) {
                    for (int d = 0; d < head_dim; d++) {
                        float sum = 0.0f;
                        
                        // Compute attention weights and apply to values
                        std::vector<float> attn_weights(seq_len_);
                        float max_score = -INFINITY;
                        
                        // Compute scores
                        for (int j = 0; j < seq_len_; j++) {
                            float score = 0.0f;
                            for (int k = 0; k < head_dim; k++) {
                                int q_idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + h * head_dim + k;
                                int k_idx = b * seq_len_ * hidden_size_ + j * hidden_size_ + h * head_dim + k;
                                score += h_Q_[q_idx] * h_K_[k_idx];
                            }
                            score *= scale;
                            attn_weights[j] = score;
                            max_score = std::max(max_score, score);
                        }
                        
                        // Softmax
                        float sum_exp = 0.0f;
                        for (int j = 0; j < seq_len_; j++) {
                            attn_weights[j] = std::exp(attn_weights[j] - max_score);
                            sum_exp += attn_weights[j];
                        }
                        for (int j = 0; j < seq_len_; j++) {
                            attn_weights[j] /= sum_exp;
                        }
                        
                        // Apply to values
                        for (int j = 0; j < seq_len_; j++) {
                            int v_idx = b * seq_len_ * hidden_size_ + j * hidden_size_ + h * head_dim + d;
                            sum += attn_weights[j] * h_V_[v_idx];
                        }
                        
                        int out_idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + h * head_dim + d;
                        h_reference_[out_idx] = sum;
                    }
                }
            }
        }
    }
    
    bool testBasicKernel() {
        std::cout << "Testing basic attention kernel..." << std::endl;
        
        // Launch kernel
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        launch_attention_kernel(d_Q_, d_K_, d_V_, d_output_, d_temp_scores_,
                               batch_size_, seq_len_, hidden_size_, num_heads_, stream);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        // Copy result back
        size_t output_size = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_output_, d_output_, output_size, cudaMemcpyDeviceToHost));
        
        // Compare with reference
        return compareResults(h_output_, h_reference_, 1e-3f);
    }
    
    bool testOptimizedKernel() {
        std::cout << "Testing optimized attention kernel..." << std::endl;
        
        // Launch kernel
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        launch_optimized_attention_kernel(d_Q_, d_K_, d_V_, d_output_, d_temp_scores_,
                                         batch_size_, seq_len_, hidden_size_, num_heads_, stream);
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        // Copy result back
        size_t output_size = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_output_, d_output_, output_size, cudaMemcpyDeviceToHost));
        
        // Compare with reference
        return compareResults(h_output_, h_reference_, 1e-3f);
    }
    
    void benchmarkKernels(int num_iterations = 100) {
        std::cout << "Benchmarking kernels..." << std::endl;
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            launch_attention_kernel(d_Q_, d_K_, d_V_, d_output_, d_temp_scores_,
                                   batch_size_, seq_len_, hidden_size_, num_heads_, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Benchmark basic kernel
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            launch_attention_kernel(d_Q_, d_K_, d_V_, d_output_, d_temp_scores_,
                                   batch_size_, seq_len_, hidden_size_, num_heads_, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        float basic_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
        
        // Benchmark optimized kernel
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            launch_optimized_attention_kernel(d_Q_, d_K_, d_V_, d_output_, d_temp_scores_,
                                             batch_size_, seq_len_, hidden_size_, num_heads_, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        end = std::chrono::high_resolution_clock::now();
        
        float optimized_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
        
        std::cout << "Basic kernel: " << basic_time << " ms" << std::endl;
        std::cout << "Optimized kernel: " << optimized_time << " ms" << std::endl;
        std::cout << "Speedup: " << basic_time / optimized_time << "x" << std::endl;
        
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
private:
    bool compareResults(const float* result, const float* reference, float tolerance) {
        int total_elements = batch_size_ * seq_len_ * hidden_size_;
        float max_diff = 0.0f;
        float rel_error = 0.0f;
        float ref_norm = 0.0f;
        float diff_norm = 0.0f;
        
        for (int i = 0; i < total_elements; i++) {
            float diff = std::abs(result[i] - reference[i]);
            max_diff = std::max(max_diff, diff);
            diff_norm += diff * diff;
            ref_norm += reference[i] * reference[i];
        }
        
        rel_error = std::sqrt(diff_norm) / std::sqrt(ref_norm);
        
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "Relative error: " << rel_error << std::endl;
        
        return max_diff < tolerance && rel_error < tolerance;
    }
};

int main() {
    std::cout << "CUDA Attention Kernel Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    
    // Test configurations
    std::vector<std::tuple<int, int, int, int>> test_configs = {
        {1, 64, 512, 8},    // Small case
        {2, 128, 768, 12},  // Medium case
        {1, 256, 1024, 16}, // Large case
    };
    
    bool all_tests_passed = true;
    
    for (const auto& config : test_configs) {
        int batch_size, seq_len, hidden_size, num_heads;
        std::tie(batch_size, seq_len, hidden_size, num_heads) = config;
        
        std::cout << "\nTesting configuration: batch=" << batch_size 
                  << ", seq_len=" << seq_len << ", hidden=" << hidden_size 
                  << ", heads=" << num_heads << std::endl;
        
        AttentionTester tester(batch_size, seq_len, hidden_size, num_heads);
        
        // Compute reference
        std::cout << "Computing CPU reference..." << std::endl;
        tester.computeReferenceAttention();
        
        // Test kernels
        bool basic_passed = tester.testBasicKernel();
        bool optimized_passed = tester.testOptimizedKernel();
        
        if (basic_passed) {
            std::cout << "✅ Basic kernel: PASSED" << std::endl;
        } else {
            std::cout << "❌ Basic kernel: FAILED" << std::endl;
            all_tests_passed = false;
        }
        
        if (optimized_passed) {
            std::cout << "✅ Optimized kernel: PASSED" << std::endl;
        } else {
            std::cout << "❌ Optimized kernel: FAILED" << std::endl;
            all_tests_passed = false;
        }
        
        // Benchmark
        tester.benchmarkKernels();
    }
    
    if (all_tests_passed) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
        return 1;
    }
    
    return 0;
}
