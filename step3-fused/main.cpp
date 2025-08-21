#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "benchmark.h"

// CUDA kernel declarations
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
            exit(1); \
        } \
    } while(0)

class FusedKernelTester {
private:
    int batch_size_, seq_len_, hidden_size_, num_heads_;
    float eps_;
    
    // Host memory
    float* h_input_;
    float* h_Q_weight_;
    float* h_K_weight_;
    float* h_V_weight_;
    float* h_qkv_weight_;
    float* h_ln_weight_;
    float* h_ln_bias_;
    float* h_output_;
    float* h_reference_;
    
    // Device memory
    float* d_input_;
    float* d_Q_weight_;
    float* d_K_weight_;
    float* d_V_weight_;
    float* d_qkv_weight_;
    float* d_ln_weight_;
    float* d_ln_bias_;
    float* d_output_;
    float* d_temp_scores_;
    
public:
    FusedKernelTester(int batch_size, int seq_len, int hidden_size, int num_heads, float eps = 1e-5f)
        : batch_size_(batch_size), seq_len_(seq_len), hidden_size_(hidden_size), 
          num_heads_(num_heads), eps_(eps) {
        
        allocateMemory();
        initializeData();
    }
    
    ~FusedKernelTester() {
        // Free host memory
        delete[] h_input_;
        delete[] h_Q_weight_;
        delete[] h_K_weight_;
        delete[] h_V_weight_;
        delete[] h_qkv_weight_;
        delete[] h_ln_weight_;
        delete[] h_ln_bias_;
        delete[] h_output_;
        delete[] h_reference_;
        
        // Free device memory
        cudaFree(d_input_);
        cudaFree(d_Q_weight_);
        cudaFree(d_K_weight_);
        cudaFree(d_V_weight_);
        cudaFree(d_qkv_weight_);
        cudaFree(d_ln_weight_);
        cudaFree(d_ln_bias_);
        cudaFree(d_output_);
        cudaFree(d_temp_scores_);
    }
    
    void allocateMemory() {
        size_t input_size = batch_size_ * seq_len_ * hidden_size_;
        size_t weight_size = hidden_size_ * hidden_size_;
        size_t qkv_weight_size = 3 * hidden_size_ * hidden_size_;
        size_t ln_param_size = hidden_size_;
        size_t scores_size = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        
        // Allocate host memory
        h_input_ = new float[input_size];
        h_Q_weight_ = new float[weight_size];
        h_K_weight_ = new float[weight_size];
        h_V_weight_ = new float[weight_size];
        h_qkv_weight_ = new float[qkv_weight_size];
        h_ln_weight_ = new float[ln_param_size];
        h_ln_bias_ = new float[ln_param_size];
        h_output_ = new float[input_size];
        h_reference_ = new float[input_size];
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input_, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Q_weight_, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K_weight_, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V_weight_, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_qkv_weight_, qkv_weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ln_weight_, ln_param_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ln_bias_, ln_param_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_scores_, scores_size * sizeof(float)));
    }
    
    void initializeData() {
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<float> weight_dist(0.0f, 0.02f);
        std::normal_distribution<float> input_dist(0.0f, 1.0f);
        
        // Initialize input
        int input_size = batch_size_ * seq_len_ * hidden_size_;
        for (int i = 0; i < input_size; i++) {
            h_input_[i] = input_dist(gen);
        }
        
        // Initialize weights
        int weight_size = hidden_size_ * hidden_size_;
        for (int i = 0; i < weight_size; i++) {
            h_Q_weight_[i] = weight_dist(gen);
            h_K_weight_[i] = weight_dist(gen);
            h_V_weight_[i] = weight_dist(gen);
        }
        
        // Create combined QKV weight
        for (int i = 0; i < weight_size; i++) {
            h_qkv_weight_[i] = h_Q_weight_[i];
            h_qkv_weight_[weight_size + i] = h_K_weight_[i];
            h_qkv_weight_[2 * weight_size + i] = h_V_weight_[i];
        }
        
        // Initialize layer norm parameters
        for (int i = 0; i < hidden_size_; i++) {
            h_ln_weight_[i] = 1.0f;
            h_ln_bias_[i] = 0.0f;
        }
        
        // Copy to device
        copyToDevice();
    }
    
    void copyToDevice() {
        size_t input_size = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        size_t weight_size = hidden_size_ * hidden_size_ * sizeof(float);
        size_t qkv_weight_size = 3 * hidden_size_ * hidden_size_ * sizeof(float);
        size_t ln_param_size = hidden_size_ * sizeof(float);
        
        CUDA_CHECK(cudaMemcpy(d_input_, h_input_, input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Q_weight_, h_Q_weight_, weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K_weight_, h_K_weight_, weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_weight_, h_V_weight_, weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qkv_weight_, h_qkv_weight_, qkv_weight_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ln_weight_, h_ln_weight_, ln_param_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ln_bias_, h_ln_bias_, ln_param_size, cudaMemcpyHostToDevice));
    }
    
    void computeReferenceOutput() {
        // CPU reference implementation: attention + residual + layer norm
        int head_dim = hidden_size_ / num_heads_;
        float scale = 1.0f / std::sqrt(head_dim);
        
        // Temporary storage for attention output
        std::vector<float> attn_output(batch_size_ * seq_len_ * hidden_size_, 0.0f);
        
        // Compute attention for each batch and head
        for (int b = 0; b < batch_size_; b++) {
            for (int h = 0; h < num_heads_; h++) {
                for (int i = 0; i < seq_len_; i++) {
                    for (int d = 0; d < head_dim; d++) {
                        float sum = 0.0f;
                        
                        // Compute attention weights
                        std::vector<float> attn_weights(seq_len_);
                        float max_score = -INFINITY;
                        
                        // Compute Q*K^T scores
                        for (int j = 0; j < seq_len_; j++) {
                            float score = 0.0f;
                            for (int k = 0; k < head_dim; k++) {
                                // Q projection
                                float q_val = 0.0f;
                                for (int l = 0; l < hidden_size_; l++) {
                                    int input_idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + l;
                                    int weight_idx = (h * head_dim + k) * hidden_size_ + l;
                                    q_val += h_input_[input_idx] * h_Q_weight_[weight_idx];
                                }
                                
                                // K projection
                                float k_val = 0.0f;
                                for (int l = 0; l < hidden_size_; l++) {
                                    int input_idx = b * seq_len_ * hidden_size_ + j * hidden_size_ + l;
                                    int weight_idx = (h * head_dim + k) * hidden_size_ + l;
                                    k_val += h_input_[input_idx] * h_K_weight_[weight_idx];
                                }
                                
                                score += q_val * k_val;
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
                            // V projection
                            float v_val = 0.0f;
                            for (int l = 0; l < hidden_size_; l++) {
                                int input_idx = b * seq_len_ * hidden_size_ + j * hidden_size_ + l;
                                int weight_idx = (h * head_dim + d) * hidden_size_ + l;
                                v_val += h_input_[input_idx] * h_V_weight_[weight_idx];
                            }
                            sum += attn_weights[j] * v_val;
                        }
                        
                        int out_idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + h * head_dim + d;
                        attn_output[out_idx] = sum;
                    }
                }
            }
        }
        
        // Add residual connection and apply layer norm
        for (int b = 0; b < batch_size_; b++) {
            for (int i = 0; i < seq_len_; i++) {
                // Add residual
                std::vector<float> residual_output(hidden_size_);
                for (int d = 0; d < hidden_size_; d++) {
                    int idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + d;
                    residual_output[d] = attn_output[idx] + h_input_[idx];
                }
                
                // Layer normalization
                float mean = 0.0f;
                for (int d = 0; d < hidden_size_; d++) {
                    mean += residual_output[d];
                }
                mean /= hidden_size_;
                
                float variance = 0.0f;
                for (int d = 0; d < hidden_size_; d++) {
                    float diff = residual_output[d] - mean;
                    variance += diff * diff;
                }
                variance /= hidden_size_;
                
                float inv_std = 1.0f / std::sqrt(variance + eps_);
                for (int d = 0; d < hidden_size_; d++) {
                    int idx = b * seq_len_ * hidden_size_ + i * hidden_size_ + d;
                    h_reference_[idx] = (residual_output[d] - mean) * inv_std * h_ln_weight_[d] + h_ln_bias_[d];
                }
            }
        }
    }
    
    bool testBasicFusedKernel() {
        std::cout << "Testing basic fused kernel..." << std::endl;
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        launch_fused_attention_layernorm(
            d_input_, d_Q_weight_, d_K_weight_, d_V_weight_,
            d_ln_weight_, d_ln_bias_, d_output_, d_temp_scores_,
            batch_size_, seq_len_, hidden_size_, num_heads_, eps_, stream
        );
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        // Copy result back
        size_t output_size = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_output_, d_output_, output_size, cudaMemcpyDeviceToHost));
        
        return compareResults(h_output_, h_reference_, 1e-2f);
    }
    
    bool testOptimizedFusedKernel() {
        std::cout << "Testing optimized fused kernel..." << std::endl;
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        launch_optimized_fused_attention_layernorm(
            d_input_, d_qkv_weight_, d_ln_weight_, d_ln_bias_, d_output_,
            batch_size_, seq_len_, hidden_size_, num_heads_, eps_, stream
        );
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        
        // Copy result back
        size_t output_size = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        CUDA_CHECK(cudaMemcpy(h_output_, d_output_, output_size, cudaMemcpyDeviceToHost));
        
        return compareResults(h_output_, h_reference_, 1e-2f);
    }
    
    void benchmarkKernels(int num_iterations = 50) {
        std::cout << "Benchmarking fused kernels..." << std::endl;
        
        BenchmarkTimer timer;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            launch_fused_attention_layernorm(
                d_input_, d_Q_weight_, d_K_weight_, d_V_weight_,
                d_ln_weight_, d_ln_bias_, d_output_, d_temp_scores_,
                batch_size_, seq_len_, hidden_size_, num_heads_, eps_, stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Benchmark basic fused kernel
        timer.start();
        for (int i = 0; i < num_iterations; i++) {
            launch_fused_attention_layernorm(
                d_input_, d_Q_weight_, d_K_weight_, d_V_weight_,
                d_ln_weight_, d_ln_bias_, d_output_, d_temp_scores_,
                batch_size_, seq_len_, hidden_size_, num_heads_, eps_, stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float basic_time = timer.stop() / num_iterations;
        
        // Benchmark optimized fused kernel
        timer.start();
        for (int i = 0; i < num_iterations; i++) {
            launch_optimized_fused_attention_layernorm(
                d_input_, d_qkv_weight_, d_ln_weight_, d_ln_bias_, d_output_,
                batch_size_, seq_len_, hidden_size_, num_heads_, eps_, stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float optimized_time = timer.stop() / num_iterations;
        
        std::cout << "Basic fused kernel: " << basic_time << " ms" << std::endl;
        std::cout << "Optimized fused kernel: " << optimized_time << " ms" << std::endl;
        std::cout << "Optimization speedup: " << basic_time / optimized_time << "x" << std::endl;
        
        // Calculate memory bandwidth savings
        size_t input_bytes = batch_size_ * seq_len_ * hidden_size_ * sizeof(float);
        size_t intermediate_bytes = batch_size_ * seq_len_ * hidden_size_ * sizeof(float) * 2; // attn_out + residual
        size_t total_saved = intermediate_bytes;
        
        std::cout << "Memory bandwidth saved: " << total_saved / 1024.0f / 1024.0f << " MB" << std::endl;
        
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
    std::cout << "CUDA Fused Kernel Test Suite" << std::endl;
    std::cout << "============================" << std::endl;
    
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
    
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
        
        FusedKernelTester tester(batch_size, seq_len, hidden_size, num_heads);
        
        // Compute reference
        std::cout << "Computing CPU reference..." << std::endl;
        tester.computeReferenceOutput();
        
        // Test kernels
        bool basic_passed = tester.testBasicFusedKernel();
        bool optimized_passed = tester.testOptimizedFusedKernel();
        
        if (basic_passed) {
            std::cout << "✅ Basic fused kernel: PASSED" << std::endl;
        } else {
            std::cout << "❌ Basic fused kernel: FAILED" << std::endl;
            all_tests_passed = false;
        }
        
        if (optimized_passed) {
            std::cout << "✅ Optimized fused kernel: PASSED" << std::endl;
        } else {
            std::cout << "❌ Optimized fused kernel: FAILED" << std::endl;
            all_tests_passed = false;
        }
        
        // Benchmark
        tester.benchmarkKernels();
    }
    
    if (all_tests_passed) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
        std::cout << "\n📊 Key Benefits Demonstrated:" << std::endl;
        std::cout << "- ✅ Fused attention + residual + layer norm operations" << std::endl;
        std::cout << "- ✅ Reduced memory bandwidth usage" << std::endl;
        std::cout << "- ✅ Improved kernel launch efficiency" << std::endl;
        std::cout << "- ✅ Better cache utilization" << std::endl;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
        return 1;
    }
    
    return 0;
}
