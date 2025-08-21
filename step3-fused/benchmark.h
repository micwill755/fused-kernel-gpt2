#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    
public:
    void start() {
        cudaDeviceSynchronize(); // Ensure all CUDA operations are complete
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    float stop() {
        cudaDeviceSynchronize(); // Ensure all CUDA operations are complete
        end_time_ = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
        return duration.count() / 1000.0f; // Return milliseconds
    }
};

class MemoryProfiler {
private:
    size_t initial_memory_;
    size_t peak_memory_;
    
public:
    void start() {
        cudaDeviceSynchronize();
        cudaMemGetInfo(&initial_memory_, nullptr);
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0); // Reset heap
    }
    
    void stop() {
        cudaDeviceSynchronize();
        size_t current_memory;
        cudaMemGetInfo(&current_memory, nullptr);
        peak_memory_ = initial_memory_ - current_memory;
    }
    
    size_t getPeakMemoryUsage() const {
        return peak_memory_;
    }
    
    void printMemoryUsage() const {
        std::cout << "Peak memory usage: " << peak_memory_ / 1024.0f / 1024.0f << " MB" << std::endl;
    }
};

class PerformanceMetrics {
public:
    struct KernelStats {
        std::string name;
        float avg_time_ms;
        float min_time_ms;
        float max_time_ms;
        float std_dev_ms;
        size_t memory_usage_mb;
        float throughput_gflops;
    };
    
    static KernelStats computeStats(const std::string& name, 
                                   const std::vector<float>& times,
                                   size_t memory_usage = 0,
                                   size_t operations = 0) {
        KernelStats stats;
        stats.name = name;
        
        // Compute timing statistics
        float sum = 0.0f;
        float min_time = times[0];
        float max_time = times[0];
        
        for (float time : times) {
            sum += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        stats.avg_time_ms = sum / times.size();
        stats.min_time_ms = min_time;
        stats.max_time_ms = max_time;
        
        // Compute standard deviation
        float variance = 0.0f;
        for (float time : times) {
            float diff = time - stats.avg_time_ms;
            variance += diff * diff;
        }
        stats.std_dev_ms = std::sqrt(variance / times.size());
        
        // Memory and throughput
        stats.memory_usage_mb = memory_usage / 1024 / 1024;
        if (operations > 0 && stats.avg_time_ms > 0) {
            stats.throughput_gflops = (operations / 1e9) / (stats.avg_time_ms / 1000.0f);
        } else {
            stats.throughput_gflops = 0.0f;
        }
        
        return stats;
    }
    
    static void printStats(const KernelStats& stats) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Kernel: " << stats.name << std::endl;
        std::cout << "  Average time: " << stats.avg_time_ms << " ms" << std::endl;
        std::cout << "  Min time: " << stats.min_time_ms << " ms" << std::endl;
        std::cout << "  Max time: " << stats.max_time_ms << " ms" << std::endl;
        std::cout << "  Std deviation: " << stats.std_dev_ms << " ms" << std::endl;
        if (stats.memory_usage_mb > 0) {
            std::cout << "  Memory usage: " << stats.memory_usage_mb << " MB" << std::endl;
        }
        if (stats.throughput_gflops > 0) {
            std::cout << "  Throughput: " << stats.throughput_gflops << " GFLOPS" << std::endl;
        }
    }
    
    static void compareKernels(const KernelStats& baseline, const KernelStats& optimized) {
        float speedup = baseline.avg_time_ms / optimized.avg_time_ms;
        float memory_reduction = 0.0f;
        if (baseline.memory_usage_mb > 0) {
            memory_reduction = (baseline.memory_usage_mb - optimized.memory_usage_mb) / 
                              (float)baseline.memory_usage_mb * 100.0f;
        }
        
        std::cout << "\nPerformance Comparison:" << std::endl;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
        if (memory_reduction != 0.0f) {
            std::cout << "  Memory reduction: " << memory_reduction << "%" << std::endl;
        }
        
        if (speedup > 1.1f) {
            std::cout << "  ✅ Significant performance improvement!" << std::endl;
        } else if (speedup > 1.0f) {
            std::cout << "  ✅ Modest performance improvement" << std::endl;
        } else {
            std::cout << "  ⚠️ Performance regression detected" << std::endl;
        }
    }
};

class GPUProfiler {
public:
    static void printGPUInfo() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            std::cout << "GPU " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Global Memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
            std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
            std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
            std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
            std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
            
            // Calculate theoretical memory bandwidth
            float memory_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
            std::cout << "  Theoretical Memory Bandwidth: " << memory_bandwidth << " GB/s" << std::endl;
        }
    }
    
    static float getMemoryBandwidthUtilization(size_t bytes_transferred, float time_ms) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        float theoretical_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // GB/s
        float actual_bandwidth = (bytes_transferred / 1.0e9) / (time_ms / 1000.0f); // GB/s
        
        return (actual_bandwidth / theoretical_bandwidth) * 100.0f; // Percentage
    }
};

// Utility functions for common benchmarking tasks
namespace BenchmarkUtils {
    
    // Run a kernel multiple times and collect timing statistics
    template<typename KernelFunc>
    std::vector<float> benchmarkKernel(KernelFunc kernel_func, int iterations = 100, int warmup = 10) {
        std::vector<float> times;
        times.reserve(iterations);
        
        BenchmarkTimer timer;
        
        // Warm up
        for (int i = 0; i < warmup; i++) {
            kernel_func();
        }
        cudaDeviceSynchronize();
        
        // Actual benchmarking
        for (int i = 0; i < iterations; i++) {
            timer.start();
            kernel_func();
            float time = timer.stop();
            times.push_back(time);
        }
        
        return times;
    }
    
    // Calculate FLOPS for attention operation
    size_t calculateAttentionFLOPS(int batch_size, int seq_len, int hidden_size, int num_heads) {
        int head_dim = hidden_size / num_heads;
        
        // Q*K^T: batch_size * num_heads * seq_len * seq_len * head_dim
        size_t qk_flops = (size_t)batch_size * num_heads * seq_len * seq_len * head_dim;
        
        // Softmax: approximately 3 * batch_size * num_heads * seq_len * seq_len
        size_t softmax_flops = 3 * (size_t)batch_size * num_heads * seq_len * seq_len;
        
        // Attention * V: batch_size * num_heads * seq_len * seq_len * head_dim
        size_t av_flops = (size_t)batch_size * num_heads * seq_len * seq_len * head_dim;
        
        return qk_flops + softmax_flops + av_flops;
    }
    
    // Calculate memory bandwidth for fused kernel
    size_t calculateFusedMemoryBandwidth(int batch_size, int seq_len, int hidden_size) {
        // Input read + output write + weights read
        size_t input_bytes = batch_size * seq_len * hidden_size * sizeof(float);
        size_t output_bytes = batch_size * seq_len * hidden_size * sizeof(float);
        size_t weight_bytes = 3 * hidden_size * hidden_size * sizeof(float); // Q, K, V weights
        size_t ln_params = 2 * hidden_size * sizeof(float); // LayerNorm weight and bias
        
        return input_bytes + output_bytes + weight_bytes + ln_params;
    }
}

#endif // BENCHMARK_H
