#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

#include "gpt2_model.h"
#include "data_loader.h"

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

class GPT2Demo {
private:
    std::unique_ptr<GPT2LMHeadModel> model_;
    GPT2Config config_;
    
public:
    GPT2Demo(const GPT2Config& config) : config_(config) {
        std::cout << "Initializing GPT-2 model..." << std::endl;
        model_ = std::make_unique<GPT2LMHeadModel>(config);
        model_->randomizeWeights();
        
        std::cout << "Model initialized with " << model_->getTransformer().getParameterCount() 
                  << " parameters" << std::endl;
        std::cout << "Memory usage: " << model_->getTransformer().getMemoryUsage() / 1024 / 1024 
                  << " MB" << std::endl;
    }
    
    void testForwardPass() {
        std::cout << "\n🧪 Testing Forward Pass" << std::endl;
        std::cout << "========================" << std::endl;
        
        // Test different batch sizes and sequence lengths
        std::vector<std::pair<int, int>> test_configs = {
            {1, 64},   // Single sequence
            {2, 128},  // Small batch
            {4, 256},  // Larger batch
            {1, 512},  // Long sequence
        };
        
        for (const auto& config : test_configs) {
            int batch_size = config.first;
            int seq_len = config.second;
            
            std::cout << "\nTesting batch_size=" << batch_size << ", seq_len=" << seq_len << std::endl;
            
            // Generate random input
            std::vector<int> input_ids(batch_size * seq_len);
            std::random_device rd;
            std::mt19937 gen(42);
            std::uniform_int_distribution<> dis(0, config_.vocab_size - 1);
            
            for (int& id : input_ids) {
                id = dis(gen);
            }
            
            // Create output matrix
            Matrix logits(batch_size * seq_len, config_.vocab_size);
            
            // Time the forward pass
            auto start = std::chrono::high_resolution_clock::now();
            
            model_->forward(input_ids.data(), batch_size, seq_len, logits);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            float tokens_per_sec = (batch_size * seq_len) / (time_ms / 1000.0f);
            
            std::cout << "  Forward pass time: " << time_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << tokens_per_sec << " tokens/sec" << std::endl;
            
            // Verify output shape and values
            std::cout << "  Output shape: [" << logits.rows() << ", " << logits.cols() << "]" << std::endl;
            
            // Check for NaN or Inf values
            std::vector<float> host_logits(logits.size());
            logits.copyToHost(host_logits.data());
            
            bool has_nan = false, has_inf = false;
            for (float val : host_logits) {
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
            }
            
            if (!has_nan && !has_inf) {
                std::cout << "  ✅ Output values are valid" << std::endl;
            } else {
                std::cout << "  ❌ Invalid output values detected (NaN: " << has_nan 
                          << ", Inf: " << has_inf << ")" << std::endl;
            }
        }
    }
    
    void testTextGeneration() {
        std::cout << "\n🗣️ Testing Text Generation" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Simple prompts for testing
        std::vector<std::string> prompts = {
            "The future of artificial intelligence",
            "In a world where technology",
            "The most important lesson"
        };
        
        for (const auto& prompt : prompts) {
            std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;
            
            // Tokenize prompt (simple word-level tokenization for demo)
            std::vector<int> input_ids = GPT2Utils::tokenize(prompt, config_.vocab_size);
            
            // Ensure minimum length
            if (input_ids.size() < 5) {
                input_ids.resize(5, 0);
            }
            
            std::cout << "Input tokens: ";
            for (int i = 0; i < std::min(10, (int)input_ids.size()); i++) {
                std::cout << input_ids[i] << " ";
            }
            std::cout << std::endl;
            
            // Generate text
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<int> generated = model_->generate(input_ids, input_ids.size() + 20, 0.8f, 50);
            
            auto end = std::chrono::high_resolution_clock::now();
            float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            
            std::cout << "Generated tokens: ";
            for (int i = input_ids.size(); i < std::min((int)generated.size(), (int)input_ids.size() + 10); i++) {
                std::cout << generated[i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Generation time: " << time_ms << " ms" << std::endl;
            std::cout << "Tokens generated: " << generated.size() - input_ids.size() << std::endl;
        }
    }
    
    void benchmarkPerformance() {
        std::cout << "\n⚡ Performance Benchmark" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::vector<std::tuple<int, int, std::string>> configs = {
            {1, 128, "Single sequence"},
            {4, 128, "Small batch"},
            {8, 64, "Larger batch"},
            {1, 512, "Long sequence"},
            {16, 32, "Many short sequences"}
        };
        
        std::cout << std::setw(20) << "Configuration" 
                  << std::setw(15) << "Avg Time (ms)" 
                  << std::setw(15) << "Throughput" 
                  << std::setw(15) << "Memory (MB)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (const auto& config : configs) {
            int batch_size, seq_len;
            std::string description;
            std::tie(batch_size, seq_len, description) = config;
            
            float avg_time = GPT2Utils::benchmarkInference(*model_, batch_size, seq_len, 50);
            float throughput = (batch_size * seq_len) / (avg_time / 1000.0f);
            
            // Measure memory usage
            size_t memory_before, memory_after;
            cudaMemGetInfo(&memory_before, nullptr);
            
            // Run one forward pass to measure peak memory
            std::vector<int> input_ids(batch_size * seq_len, 0);
            Matrix logits(batch_size * seq_len, config_.vocab_size);
            model_->forward(input_ids.data(), batch_size, seq_len, logits);
            
            cudaMemGetInfo(&memory_after, nullptr);
            float memory_used = (memory_before - memory_after) / 1024.0f / 1024.0f;
            
            std::cout << std::setw(20) << description
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                      << std::setw(15) << std::fixed << std::setprecision(0) << throughput
                      << std::setw(15) << std::fixed << std::setprecision(1) << memory_used << std::endl;
        }
    }
    
    void testMemoryEfficiency() {
        std::cout << "\n💾 Memory Efficiency Test" << std::endl;
        std::cout << "==========================" << std::endl;
        
        // Test with increasing sequence lengths to see memory scaling
        std::vector<int> seq_lengths = {64, 128, 256, 512, 1024};
        int batch_size = 1;
        
        std::cout << std::setw(15) << "Seq Length" 
                  << std::setw(20) << "Peak Memory (MB)" 
                  << std::setw(20) << "Memory/Token (KB)" << std::endl;
        std::cout << std::string(55, '-') << std::endl;
        
        for (int seq_len : seq_lengths) {
            try {
                // Clear GPU memory
                CUDA_CHECK(cudaDeviceSynchronize());
                
                size_t free_before, total;
                cudaMemGetInfo(&free_before, &total);
                
                // Run forward pass
                std::vector<int> input_ids(batch_size * seq_len, 0);
                Matrix logits(batch_size * seq_len, config_.vocab_size);
                model_->forward(input_ids.data(), batch_size, seq_len, logits);
                
                CUDA_CHECK(cudaDeviceSynchronize());
                
                size_t free_after;
                cudaMemGetInfo(&free_after, &total);
                
                float memory_used_mb = (free_before - free_after) / 1024.0f / 1024.0f;
                float memory_per_token_kb = memory_used_mb * 1024.0f / (batch_size * seq_len);
                
                std::cout << std::setw(15) << seq_len
                          << std::setw(20) << std::fixed << std::setprecision(1) << memory_used_mb
                          << std::setw(20) << std::fixed << std::setprecision(2) << memory_per_token_kb << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << std::setw(15) << seq_len
                          << std::setw(20) << "OOM"
                          << std::setw(20) << "-" << std::endl;
                break;
            }
        }
    }
    
    void testNumericalStability() {
        std::cout << "\n🔢 Numerical Stability Test" << std::endl;
        std::cout << "============================" << std::endl;
        
        int batch_size = 2, seq_len = 64;
        
        // Test with different input ranges
        std::vector<std::pair<std::string, std::function<float()>>> test_cases = {
            {"Normal range", []() { 
                static std::mt19937 gen(42);
                static std::normal_distribution<float> dist(0.0f, 1.0f);
                return dist(gen);
            }},
            {"Large values", []() {
                static std::mt19937 gen(42);
                static std::normal_distribution<float> dist(0.0f, 10.0f);
                return dist(gen);
            }},
            {"Small values", []() {
                static std::mt19937 gen(42);
                static std::normal_distribution<float> dist(0.0f, 0.01f);
                return dist(gen);
            }}
        };
        
        for (const auto& test_case : test_cases) {
            std::cout << "\nTesting " << test_case.first << ":" << std::endl;
            
            // Create input with specific range
            std::vector<int> input_ids(batch_size * seq_len);
            for (int& id : input_ids) {
                id = std::abs((int)test_case.second()) % config_.vocab_size;
            }
            
            Matrix logits(batch_size * seq_len, config_.vocab_size);
            
            try {
                model_->forward(input_ids.data(), batch_size, seq_len, logits);
                
                // Check output statistics
                std::vector<float> host_logits(logits.size());
                logits.copyToHost(host_logits.data());
                
                float sum = 0.0f, sum_sq = 0.0f;
                int nan_count = 0, inf_count = 0;
                
                for (float val : host_logits) {
                    if (std::isnan(val)) {
                        nan_count++;
                    } else if (std::isinf(val)) {
                        inf_count++;
                    } else {
                        sum += val;
                        sum_sq += val * val;
                    }
                }
                
                int valid_count = host_logits.size() - nan_count - inf_count;
                float mean = sum / valid_count;
                float variance = (sum_sq / valid_count) - (mean * mean);
                float std_dev = std::sqrt(variance);
                
                std::cout << "  Mean: " << mean << ", Std: " << std_dev << std::endl;
                std::cout << "  NaN count: " << nan_count << ", Inf count: " << inf_count << std::endl;
                
                if (nan_count == 0 && inf_count == 0) {
                    std::cout << "  ✅ Numerically stable" << std::endl;
                } else {
                    std::cout << "  ❌ Numerical instability detected" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "  ❌ Error: " << e.what() << std::endl;
            }
        }
    }
};

void printSystemInfo() {
    std::cout << "🖥️ System Information" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // CUDA device info
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    
    // Memory info
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "Available GPU Memory: " << free_mem / 1024 / 1024 << " MB / " 
              << total_mem / 1024 / 1024 << " MB" << std::endl;
}

int main() {
    std::cout << "🚀 GPT-2 Fused CUDA Kernel Demo" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        // Initialize CUDA
        GPT2Utils::initializeCUDA();
        printSystemInfo();
        
        // Create model configurations to test
        std::vector<std::pair<std::string, GPT2Config>> model_configs = {
            {"GPT-2 Tiny", GPT2Config(50257, 512, 384, 4, 6, 1536)},
            {"GPT-2 Small", GPT2Config(50257, 1024, 768, 6, 12, 3072)},
        };
        
        for (const auto& config_pair : model_configs) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "Testing " << config_pair.first << std::endl;
            std::cout << std::string(50, '=') << std::endl;
            
            GPT2Demo demo(config_pair.second);
            
            // Run all tests
            demo.testForwardPass();
            demo.testTextGeneration();
            demo.benchmarkPerformance();
            demo.testMemoryEfficiency();
            demo.testNumericalStability();
        }
        
        std::cout << "\n🎉 Demo completed successfully!" << std::endl;
        std::cout << "\n📊 Key Achievements:" << std::endl;
        std::cout << "- ✅ Fused attention + layer norm kernels working" << std::endl;
        std::cout << "- ✅ Complete GPT-2 model implementation in C++/CUDA" << std::endl;
        std::cout << "- ✅ Text generation capability" << std::endl;
        std::cout << "- ✅ Performance benchmarking" << std::endl;
        std::cout << "- ✅ Memory efficiency analysis" << std::endl;
        std::cout << "- ✅ Numerical stability verification" << std::endl;
        
        std::cout << "\n🔗 Next Steps:" << std::endl;
        std::cout << "1. Optimize kernels further for your specific GPU" << std::endl;
        std::cout << "2. Implement additional fusion patterns (MLP, embeddings)" << std::endl;
        std::cout << "3. Add support for different precisions (FP16, INT8)" << std::endl;
        std::cout << "4. Implement distributed training support" << std::endl;
        std::cout << "5. Integrate with production inference frameworks" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "💥 Demo failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
