# Step 1: Simple CUDA Kernel

## Goal
Learn the basics of CUDA kernel development by implementing a simple element-wise addition operation.

## What You'll Learn
- Basic CUDA kernel structure
- Thread indexing and boundary checking
- PyTorch C++ extensions
- CUDA memory management basics
- Performance benchmarking

## Files in This Step
- `simple_kernel.cu` - The CUDA kernel implementation
- `setup.py` - Build configuration
- `test.py` - Test and benchmark script

## How to Run

1. **Build the extension**:
   ```bash
   cd step1-simple
   python setup.py build_ext --inplace
   ```

2. **Run the test**:
   ```bash
   python test.py
   ```

## Expected Output
```
🚀 Testing Simple CUDA Kernel (Step 1)
==================================================
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 11.8

🧪 Testing kernel correctness...
✅ Correctness test PASSED!

⚡ Benchmarking performance...

Testing size: 100x100 = 10,000 elements
  Our CUDA kernel: 0.045 ms
  PyTorch:         0.032 ms
  Speedup:         0.71x

Testing size: 1000x1000 = 1,000,000 elements
  Our CUDA kernel: 0.156 ms
  PyTorch:         0.145 ms
  Speedup:         0.93x

Testing size: 5000x5000 = 25,000,000 elements
  Our CUDA kernel: 3.421 ms
  PyTorch:         3.398 ms
  Speedup:         0.99x

🎉 All tests completed successfully!
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

## Performance Notes

Don't worry if your kernel is slightly slower than PyTorch for simple operations. PyTorch's kernels are highly optimized. The real benefits come with more complex operations and kernel fusion.

## Troubleshooting

**Build Errors**:
- Check CUDA installation: `nvcc --version`
- Verify PyTorch CUDA support: `torch.cuda.is_available()`
- Check GPU compute capability matches setup.py

**Runtime Errors**:
- Add error checking after kernel launch
- Use `cuda-gdb` for debugging
- Check tensor types and devices

## Next Step

Once this works correctly, proceed to `step2-attention/` to implement a more complex attention kernel.

## Key Takeaways

1. CUDA kernels run many threads in parallel
2. Each thread processes one or more elements
3. Always check array bounds
4. PyTorch integration is straightforward with pybind11
5. Simple kernels may not beat PyTorch's optimized implementations
