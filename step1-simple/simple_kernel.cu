// Step 1: Simple CUDA Kernel - Element-wise Addition
// Compile directly with: nvcc -o simple_kernel simple_kernel.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// CUDA kernel function - runs on GPU
// A kernel is simply parallel code that runs on the GPU
__global__ void simple_add_kernel(
    const float* input_a,    // Input tensor A - in this example tensors are only a 1d array, but in future
                             // examples these will become multi-dimensional tensors - GPT-2 operations: attention matrices, embeddings, layer outputs
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

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Initialize array with random values
void init_array(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

// Verify results by comparing with CPU computation
bool verify_results(const float* a, const float* b, const float* gpu_result, int size) {
    const float tolerance = 1e-5f;
    
    for (int i = 0; i < size; i++) {
        float expected = a[i] + b[i];
        float diff = fabsf(gpu_result[i] - expected);
        
        if (diff > tolerance) {
            printf("Verification failed at index %d: expected %f, got %f (diff: %f)\n", 
                   i, expected, gpu_result[i], diff);
            return false;
        }
    }
    return true;
}

// Benchmark function
double benchmark_kernel(float* d_a, float* d_b, float* d_output, int size, int iterations) {
    const int threads_per_block = 256;
    const int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    // Warm up - we create 10 identical runs of the same simple_add_kernel that will be timed later.
    for (int i = 0; i < 10; i++) {
        simple_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_output, size);
    }
    // cudaDeviceSynchronize() forces the CPU to wait for all GPU work to complete
    CUDA_CHECK(cudaDeviceSynchronize()); 
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        simple_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_output, size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds / iterations; // Average time per iteration
}

int main() {
    printf("Simple CUDA Kernel - Element-wise Addition\n");
    printf("==============================================\n");
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    
    // Test different array sizes
    int sizes[] = {10000, 1000000, 25000000}; // 10K, 1M, 25M elements
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    srand(time(NULL)); // Initialize random seed
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        size_t bytes = size * sizeof(float);
        
        printf("\n Testing with %d elements (%.2f MB)\n", size, bytes / (1024.0f * 1024.0f));
        
        // Allocate host memory
        float* h_a = (float*)malloc(bytes);
        float* h_b = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        
        if (!h_a || !h_b || !h_output) {
            printf("Failed to allocate host memory\n");
            return 1;
        }
        
        // Initialize input arrays
        init_array(h_a, size);
        init_array(h_b, size);
        
        // Allocate device memory
        float* d_a;
        float* d_b;
        float* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        
        // Launch kernel
        const int threads_per_block = 256;
        const int blocks = (size + threads_per_block - 1) / threads_per_block;
        
        printf("Launching kernel: %d blocks × %d threads = %d total threads\n", 
               blocks, threads_per_block, blocks * threads_per_block);
        
        // The <<<>>> syntax launches the kernel on the GPU - syntax can take up to 4 parameters, but only the first 2 are required:
        // e.g. kernel <<<grid_size, block_size, shared_mem, stream>>>(args);
        // grid_size (required): Number of thread blocks
        // block_size (required): Threads per block
        // shared_mem (optional): Shared memory bytes per block (default: 0)
        // stream (optional): CUDA stream for async execution (default: 0)
        simple_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_output, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Verify results
        if (verify_results(h_a, h_b, h_output, size)) {
            printf("Correctness test PASSED!\n");
        } else {
            printf("Correctness test FAILED!\n");
            return 1;
        }
        
        // Benchmark performance
        double avg_time = benchmark_kernel(d_a, d_b, d_output, size, 100);
        double bandwidth = (3.0 * bytes) / (avg_time * 1e-3) / 1e9; // GB/s (3 arrays: 2 reads + 1 write)
        
        printf("⚡ Performance: %.3f ms (%.2f GB/s)\n", avg_time, bandwidth);
        
        // Cleanup
        free(h_a);
        free(h_b);
        free(h_output);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_output));
    }
    
    printf("\nAll tests completed successfully!\n");

    return 0;
}
