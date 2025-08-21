# Debugging CUDA Kernels

## Overview
Debugging CUDA kernels can be challenging due to the parallel nature of GPU execution. This guide provides comprehensive techniques and tools for debugging CUDA code effectively.

## Common CUDA Errors

### Runtime Errors

#### CUDA_ERROR_INVALID_VALUE
```cuda
// Common cause: Invalid grid/block dimensions
kernel<<<0, 256>>>();  // gridDim.x cannot be 0
kernel<<<256, 0>>>();  // blockDim.x cannot be 0
kernel<<<256, 1025>>>(); // blockDim.x too large (max 1024)
```

#### CUDA_ERROR_OUT_OF_MEMORY
```cuda
// Check available memory
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("Free: %zu MB, Total: %zu MB\n", 
       free_mem/1024/1024, total_mem/1024/1024);

// Reduce memory usage
// - Smaller batch sizes
// - Lower precision (FP16)
// - Memory pooling
```

#### CUDA_ERROR_LAUNCH_FAILED
```cuda
// Usually indicates kernel crash
// Check for:
// - Array bounds violations
// - Null pointer dereferences
// - Division by zero
// - Invalid memory access
```

### Compilation Errors

#### Undefined Symbol Errors
```bash
# Missing CUDA libraries
nvcc -lcudart -lcublas kernel.cu

# Wrong architecture
nvcc -arch=sm_70 kernel.cu  # For V100
nvcc -arch=sm_86 kernel.cu  # For RTX 30xx
```

## Error Checking Techniques

### Comprehensive Error Checking Macro
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            fprintf(stderr, "Error code: %d\n", error); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
kernel<<<grid, block>>>(d_data);
CUDA_CHECK(cudaGetLastError());  // Check kernel launch
CUDA_CHECK(cudaDeviceSynchronize());  // Check kernel execution
```

### Kernel Launch Error Checking
```cuda
// Check kernel launch parameters
void check_launch_params(dim3 grid, dim3 block, size_t shared_mem) {
    // Check block dimensions
    if (block.x * block.y * block.z > 1024) {
        fprintf(stderr, "Block size too large: %d\n", 
                block.x * block.y * block.z);
        exit(1);
    }
    
    // Check shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem > prop.sharedMemPerBlock) {
        fprintf(stderr, "Shared memory too large: %zu > %zu\n", 
                shared_mem, prop.sharedMemPerBlock);
        exit(1);
    }
}
```

## Debugging Tools

### CUDA-GDB
```bash
# Compile with debug info
nvcc -g -G -O0 kernel.cu -o program

# Run with cuda-gdb
cuda-gdb ./program

# GDB commands for CUDA
(cuda-gdb) set cuda memcheck on
(cuda-gdb) run
(cuda-gdb) cuda kernel
(cuda-gdb) cuda block
(cuda-gdb) cuda thread
(cuda-gdb) print variable_name
```

### CUDA-MEMCHECK
```bash
# Check for memory errors
cuda-memcheck ./program

# Common errors detected:
# - Out of bounds memory access
# - Uninitialized memory usage
# - Memory leaks
# - Race conditions
```

### Compute Sanitizer (CUDA 11.6+)
```bash
# More advanced memory checking
compute-sanitizer --tool memcheck ./program
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool synccheck ./program
```

### NVIDIA Nsight Compute
```bash
# Profile kernel performance
nsight-compute ./program

# Specific metrics
nsight-compute --metrics sm__cycles_elapsed.avg,dram__bytes_read.sum ./program

# Interactive profiling
nsight-compute-cli --mode=launch --launch-skip-before-match=0 ./program
```

## In-Kernel Debugging

### Printf Debugging
```cuda
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Debug specific threads
    if (idx == 0) {
        printf("Block %d, Thread %d: data[0] = %f\n", 
               blockIdx.x, threadIdx.x, data[0]);
    }
    
    // Debug problematic values
    if (data[idx] != data[idx]) {  // Check for NaN
        printf("NaN detected at index %d\n", idx);
    }
    
    if (isinf(data[idx])) {  // Check for infinity
        printf("Inf detected at index %d\n", idx);
    }
}
```

### Conditional Debugging
```cuda
#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

__global__ void kernel(float* data) {
    int idx = threadIdx.x;
    DEBUG_PRINT("Thread %d processing data[%d] = %f\n", idx, idx, data[idx]);
}
```

### Assert in Kernels
```cuda
#include <assert.h>

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Runtime assertions
    assert(idx < n);
    assert(data != nullptr);
    assert(data[idx] >= 0.0f);  // Check valid range
}
```

## Memory Debugging

### Bounds Checking
```cuda
__device__ float safe_access(float* array, int idx, int size) {
    if (idx < 0 || idx >= size) {
        printf("Out of bounds access: idx=%d, size=%d\n", idx, size);
        return 0.0f;
    }
    return array[idx];
}

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = safe_access(data, idx, n);
}
```

### Memory Pattern Detection
```cuda
// Fill memory with pattern to detect corruption
void fill_debug_pattern(float* ptr, size_t count) {
    for (size_t i = 0; i < count; i++) {
        ptr[i] = (float)(0xDEADBEEF + i);
    }
}

// Check if pattern is intact
bool check_debug_pattern(float* ptr, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (ptr[i] != (float)(0xDEADBEEF + i)) {
            printf("Memory corruption at index %zu\n", i);
            return false;
        }
    }
    return true;
}
```

### Shared Memory Debugging
```cuda
__global__ void debug_shared_memory() {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    
    // Initialize shared memory
    shared_data[tid] = tid;
    __syncthreads();
    
    // Check for bank conflicts
    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            if (shared_data[i] != i) {
                printf("Shared memory corruption at index %d\n", i);
            }
        }
    }
}
```

## Performance Debugging

### Timing Individual Operations
```cuda
__global__ void timed_kernel(float* data, int n) {
    clock_t start = clock();
    
    // Your kernel code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
    
    clock_t end = clock();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel execution time: %d cycles\n", (int)(end - start));
    }
}
```

### Occupancy Analysis
```cuda
// Check theoretical occupancy
void check_occupancy() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int block_size = 256;
    int min_grid_size, grid_size;
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                       kernel, 0, 0);
    
    printf("Suggested block size: %d\n", block_size);
    printf("Minimum grid size: %d\n", min_grid_size);
}
```

## Debugging Strategies

### Divide and Conquer
```cuda
// Test components separately
__global__ void test_component_1(float* input, float* output) {
    // Test only the first part of your algorithm
}

__global__ void test_component_2(float* input, float* output) {
    // Test only the second part
}

// Then combine when both work correctly
```

### Simplify Input Data
```cuda
// Use simple, predictable input
void create_debug_input(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;  // Simple pattern
    }
}

// Expected output should be easy to verify
```

### Compare with CPU Implementation
```cuda
// CPU reference implementation
void cpu_reference(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = sqrtf(input[i]);
    }
}

// Compare results
bool compare_results(float* gpu_result, float* cpu_result, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", 
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}
```

## Advanced Debugging Techniques

### Warp-Level Debugging
```cuda
__global__ void debug_warp_behavior() {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Debug warp divergence
    if (threadIdx.x % 2 == 0) {
        // Even threads
        printf("Warp %d, Lane %d: Even branch\n", warp_id, lane_id);
    } else {
        // Odd threads - causes divergence
        printf("Warp %d, Lane %d: Odd branch\n", warp_id, lane_id);
    }
}
```

### Race Condition Detection
```cuda
__global__ void detect_race_conditions() {
    __shared__ int counter;
    
    if (threadIdx.x == 0) {
        counter = 0;
    }
    __syncthreads();
    
    // Potential race condition
    int old_val = atomicAdd(&counter, 1);
    
    // Check for unexpected behavior
    if (old_val >= blockDim.x) {
        printf("Race condition detected: counter=%d\n", old_val);
    }
}
```

### Memory Access Pattern Visualization
```cuda
__global__ void visualize_memory_access(float* data, int* access_pattern) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Record access pattern
    access_pattern[threadIdx.x] = idx;
    
    // Process data
    data[idx] = data[idx] * 2.0f;
}

// Analyze access pattern on host
void analyze_access_pattern(int* pattern, int size) {
    printf("Memory access pattern:\n");
    for (int i = 0; i < size; i++) {
        printf("Thread %d accesses index %d\n", i, pattern[i]);
    }
}
```

## Debugging Checklist

### Before Debugging
- [ ] Reproduce the issue consistently
- [ ] Identify the specific kernel causing problems
- [ ] Gather error messages and symptoms
- [ ] Create minimal test case

### During Debugging
- [ ] Add comprehensive error checking
- [ ] Use appropriate debugging tools
- [ ] Test with simple, known inputs
- [ ] Compare with CPU reference
- [ ] Check memory access patterns

### After Debugging
- [ ] Verify fix with multiple test cases
- [ ] Check performance impact of debug code
- [ ] Remove debug prints from production code
- [ ] Document the issue and solution

## Common Debugging Scenarios

### Scenario 1: Kernel Produces Wrong Results
1. Add printf statements to check intermediate values
2. Compare with CPU reference implementation
3. Test with simple input data
4. Check array bounds and indexing logic

### Scenario 2: Kernel Crashes
1. Use cuda-memcheck to detect memory errors
2. Add bounds checking to array accesses
3. Check for null pointer dereferences
4. Verify kernel launch parameters

### Scenario 3: Poor Performance
1. Profile with nsight-compute
2. Check memory access patterns
3. Analyze occupancy
4. Look for warp divergence

### Scenario 4: Intermittent Failures
1. Check for race conditions
2. Verify synchronization points
3. Test with different input sizes
4. Use compute-sanitizer race checker

## Best Practices

### Development Workflow
1. **Start Simple**: Begin with basic functionality
2. **Test Incrementally**: Add complexity gradually
3. **Use Version Control**: Track working versions
4. **Document Issues**: Keep notes on problems and solutions

### Code Organization
```cuda
// Separate debug and release builds
#ifdef DEBUG
    #define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
    #define SYNC_CHECK() CUDA_CHECK(cudaDeviceSynchronize())
#else
    #define KERNEL_CHECK()
    #define SYNC_CHECK()
#endif

kernel<<<grid, block>>>(args);
KERNEL_CHECK();
SYNC_CHECK();
```

### Testing Strategy
- Unit test individual kernels
- Integration test complete workflows
- Stress test with large inputs
- Regression test after changes
