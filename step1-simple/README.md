# Step 1: Simple CUDA Kernel

## Goal
Learn the basics of CUDA kernel development by implementing a simple element-wise addition operation using direct `nvcc` compilation.

## What You'll Learn
- Basic CUDA kernel structure
- Thread indexing and boundary checking
- Direct CUDA compilation with `nvcc`
- CUDA memory management basics
- Performance benchmarking
- Error handling in CUDA

## Files in This Step
- `simple_kernel.cu` - The standalone CUDA C++ implementation
- `setup.py` - Original PyTorch build configuration (for reference)
- `test.py` - Original PyTorch test script (for reference)

## How to Run

### Simple Compilation (Recommended)
```bash
cd step1-simple
nvcc -o simple_kernel simple_kernel.cu
./simple_kernel
```

### If you get driver/runtime version mismatch
Use the CUDA version that matches your driver:
```bash
# Check your driver's CUDA version with: nvidia-smi
# Then use the matching CUDA toolkit:
/usr/local/cuda-12.8/bin/nvcc -o simple_kernel simple_kernel.cu
./simple_kernel
```

### With specific GPU architecture (optional)
```bash
# For A10G/RTX 30xx series (Ampere)
nvcc -o simple_kernel simple_kernel.cu --generate-code arch=compute_86,code=sm_86

# For RTX 40xx series (Ada Lovelace)  
nvcc -o simple_kernel simple_kernel.cu --generate-code arch=compute_89,code=sm_89
```

## Expected Output
```
🚀 Simple CUDA Kernel - Element-wise Addition
==============================================
GPU: NVIDIA A10G
Compute Capability: 8.6
Max Threads Per Block: 1024

🧪 Testing with 10000 elements (0.04 MB)
Launching kernel: 40 blocks × 256 threads = 10240 total threads
✅ Correctness test PASSED!
⚡ Performance: 0.003 ms (44.56 GB/s)

🧪 Testing with 1000000 elements (3.81 MB)
Launching kernel: 3907 blocks × 256 threads = 1000192 total threads
✅ Correctness test PASSED!
⚡ Performance: 0.024 ms (491.97 GB/s)

🧪 Testing with 25000000 elements (95.37 MB)
Launching kernel: 97657 blocks × 256 threads = 25000192 total threads
✅ Correctness test PASSED!
⚡ Performance: 0.607 ms (494.37 GB/s)

🎉 All tests completed successfully!

💡 To compile and run:
   nvcc -o simple_kernel simple_kernel.cu
   ./simple_kernel
```

## Understanding the Code

### CUDA Kernel Structure
```cuda
__global__ void simple_add_kernel(
    const float* input_a,    // Input pointer
    const float* input_b,    // Input pointer
    float* output,           // Output pointer
    int num_elements         // Size parameter
) {
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx < num_elements) {
        output[idx] = input_a[idx] + input_b[idx];
    }
}
```

### Key Concepts

**Thread Indexing**:
- `blockIdx.x`: Which block this thread belongs to
- `blockDim.x`: Number of threads per block
- `threadIdx.x`: Thread index within the block
- `idx = blockIdx.x * blockDim.x + threadIdx.x`: Global thread index

**Launch Configuration**:
```cuda
simple_add_kernel<<<blocks, threads_per_block>>>(...)
```
- `blocks`: Number of thread blocks
- `threads_per_block`: Threads per block (usually 256 or 512)

**Memory Management**:
```cpp
// Allocate GPU memory
cudaMalloc(&d_ptr, size_in_bytes);

// Copy data to GPU
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// Copy data from GPU
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_ptr);
```

**Error Checking**:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

## Performance Notes

The program tests three different array sizes and measures:
- **Correctness**: Verifies GPU results match CPU computation
- **Performance**: Measures kernel execution time
- **Bandwidth**: Calculates memory bandwidth utilization

For simple element-wise operations like addition, performance is typically memory-bound rather than compute-bound. The A10G achieves ~490 GB/s which is close to its theoretical memory bandwidth.

## Troubleshooting

**Compilation Errors**:
- Check CUDA installation: `nvcc --version`
- Verify GPU compute capability: `nvidia-smi`
- Use appropriate CUDA version that matches your driver

**Runtime Errors**:
- **Driver version mismatch**: Use matching CUDA toolkit version (e.g., `/usr/local/cuda-12.8/bin/nvcc`)
- **Out of memory**: Reduce array sizes in the code
- **No CUDA devices**: Check `nvidia-smi` output

**GPU Architecture Flags**:
- A10G/RTX 30xx: `--generate-code arch=compute_86,code=sm_86`
- RTX 40xx: `--generate-code arch=compute_89,code=sm_89`
- RTX 20xx: `--generate-code arch=compute_75,code=sm_75`

## Advantages of Direct nvcc Compilation

1. **No dependencies**: Pure C++/CUDA, no Python/PyTorch needed
2. **No version conflicts**: Works with any compatible CUDA/driver combination
3. **Fast compilation**: Direct compilation without build system overhead
4. **Better debugging**: Direct access to CUDA debugging tools (`cuda-gdb`)
5. **Educational**: Clearer understanding of CUDA fundamentals
6. **Portable**: Single source file, easy to share and modify

## Next Steps

This approach can be extended to more complex kernels:
- Matrix multiplication
- Convolution operations
- Attention mechanisms
- Custom fused operations

The same compilation pattern works for any CUDA kernel:
```bash
nvcc -o my_kernel my_kernel.cu
./my_kernel
```

## Key Takeaways

1. CUDA kernels run many threads in parallel
2. Each thread processes one or more elements
3. Always check array bounds and CUDA errors
4. Memory bandwidth often limits performance for simple operations
5. Direct nvcc compilation avoids framework dependencies
6. Proper error handling is crucial for CUDA development
7. Use the CUDA toolkit version that matches your driver
