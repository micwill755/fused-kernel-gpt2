# Memory Optimization Techniques

## Overview
Memory optimization is crucial for CUDA kernel performance. This guide covers techniques to minimize memory usage and maximize memory bandwidth utilization.

## Memory Hierarchy Review

### GPU Memory Types (Speed: Fast → Slow)
1. **Registers**: 32KB per SM, fastest access
2. **Shared Memory**: 48-164KB per SM, user-controlled cache
3. **L1 Cache**: 128KB per SM, automatic caching
4. **L2 Cache**: 6-40MB, shared across SMs
5. **Global Memory**: GBs, highest latency

## Memory Access Patterns

### Coalesced Access
**Good Pattern**: Consecutive threads access consecutive memory locations
```cuda
// Thread 0 accesses data[0], thread 1 accesses data[1], etc.
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = data[idx];
```

**Bad Pattern**: Strided or random access
```cuda
// Large stride - poor coalescing
int idx = threadIdx.x * 1024;
float value = data[idx];
```

### Memory Coalescing Rules
- **128-byte transactions**: GPU loads memory in 128-byte chunks
- **Alignment**: Starting address should be aligned to 128 bytes
- **Consecutive access**: Threads in a warp should access consecutive addresses

## Shared Memory Optimization

### Bank Conflicts
Shared memory is organized into 32 banks. Avoid multiple threads accessing the same bank simultaneously.

```cuda
__shared__ float shared_data[32][33]; // +1 to avoid bank conflicts

// Good: No bank conflicts
int tid = threadIdx.x;
shared_data[tid][0] = input[tid];

// Bad: Bank conflicts
shared_data[0][tid] = input[tid]; // All threads access bank 0
```

### Shared Memory Usage Patterns

#### Tiled Matrix Multiplication
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Load tile collaboratively
As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
__syncthreads();

// Compute using shared memory
for (int k = 0; k < TILE_SIZE; ++k) {
    sum += As[ty][k] * Bs[k][tx];
}
```

#### Reduction in Shared Memory
```cuda
__shared__ float sdata[256];

// Load data
sdata[tid] = input[blockIdx.x * blockDim.x + tid];
__syncthreads();

// Tree reduction
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

## Register Optimization

### Minimize Register Usage
High register usage reduces occupancy. Use these techniques:

```cuda
// Bad: Creates many temporary variables
float temp1 = a * b;
float temp2 = c * d;
float temp3 = temp1 + temp2;
float result = temp3 * e;

// Better: Reuse variables
float temp = a * b;
temp += c * d;
temp *= e;
```

### Register Spilling
When kernels use too many registers, they "spill" to local memory (slow).

```bash
# Check register usage during compilation
nvcc -Xptxas=-v kernel.cu
# Output: ptxas info : Used 32 registers, 384 bytes cmem[0]
```

### Occupancy vs Register Usage
```cuda
// Limit registers for better occupancy
__global__ void __launch_bounds__(256, 4) kernel() {
    // 256 threads per block, minimum 4 blocks per SM
}
```

## Memory Access Optimization

### Vectorized Memory Access
Use vector types for better memory throughput:

```cuda
// Load 4 floats at once
float4 data = reinterpret_cast<float4*>(input)[idx];
float a = data.x, b = data.y, c = data.z, d = data.w;

// Store 4 floats at once
float4 result = make_float4(a, b, c, d);
reinterpret_cast<float4*>(output)[idx] = result;
```

### Memory Prefetching
```cuda
// Prefetch next iteration's data
__shared__ float shared_current[TILE_SIZE];
__shared__ float shared_next[TILE_SIZE];

for (int iter = 0; iter < num_iterations; ++iter) {
    // Load next iteration's data while computing current
    if (iter + 1 < num_iterations) {
        shared_next[tid] = input[(iter + 1) * blockDim.x + tid];
    }
    
    // Compute using current data
    float result = compute(shared_current[tid]);
    
    // Swap buffers
    __syncthreads();
    float* temp = shared_current;
    shared_current = shared_next;
    shared_next = temp;
}
```

## Kernel Fusion Benefits

### Memory Bandwidth Reduction
Fusing operations reduces intermediate memory traffic:

```cuda
// Separate kernels: 3 global memory accesses per element
kernel1<<<...>>>(input, temp1);    // Read input, write temp1
kernel2<<<...>>>(temp1, temp2);    // Read temp1, write temp2
kernel3<<<...>>>(temp2, output);   // Read temp2, write output

// Fused kernel: 1 read + 1 write per element
fused_kernel<<<...>>>(input, output); // Read input, write output
```

### Cache Efficiency
Fused kernels keep intermediate results in registers/shared memory:

```cuda
__global__ void fused_attention_layernorm(input, output) {
    // Load input once
    float x = input[idx];
    
    // Attention computation (stays in registers)
    float attn_out = attention_compute(x);
    
    // Add residual (no memory access)
    float residual = attn_out + x;
    
    // Layer norm (minimal memory access)
    float normalized = layer_norm(residual);
    
    // Write output once
    output[idx] = normalized;
}
```

## Memory Pool Management

### Custom Memory Allocators
```cuda
class CUDAMemoryPool {
private:
    std::vector<void*> free_blocks;
    std::map<size_t, std::vector<void*>> size_to_blocks;
    
public:
    void* allocate(size_t size) {
        // Try to reuse existing block
        auto it = size_to_blocks.find(size);
        if (it != size_to_blocks.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        
        // Allocate new block
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void deallocate(void* ptr, size_t size) {
        size_to_blocks[size].push_back(ptr);
    }
};
```

### Memory Pinning
```cuda
// Pinned memory for faster CPU-GPU transfers
float* host_data;
cudaMallocHost(&host_data, size);  // Pinned memory

// Faster than regular malloc
cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
```

## Profiling Memory Usage

### NVIDIA Profiler Commands
```bash
# Profile memory usage
nsight-compute --metrics dram__bytes_read,dram__bytes_write ./program

# Memory bandwidth utilization
nsight-compute --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Cache hit rates
nsight-compute --metrics l1tex__t_sector_hit_rate,l2_t_sector_hit_rate ./program
```

### In-Code Memory Monitoring
```cuda
// Check memory usage
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("GPU Memory: %zu MB free, %zu MB total\n", 
       free_mem / 1024 / 1024, total_mem / 1024 / 1024);
```

## Advanced Techniques

### Memory Compression
```cuda
// Use half precision where possible
__half* half_data;
float* float_data;

// Convert for computation
float value = __half2float(half_data[idx]);
// ... compute ...
half_data[idx] = __float2half(result);
```

### Asynchronous Memory Transfers
```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap computation and memory transfer
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(d_data, d_output);
cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream1);
```

### Unified Memory
```cuda
// Unified memory (CUDA 6.0+)
float* unified_data;
cudaMallocManaged(&unified_data, size);

// Accessible from both CPU and GPU
unified_data[0] = 1.0f;  // CPU access
kernel<<<...>>>(unified_data);  // GPU access
```

## Best Practices Summary

### Do's
- ✅ Use coalesced memory access patterns
- ✅ Minimize global memory transactions
- ✅ Use shared memory for data reuse
- ✅ Avoid bank conflicts in shared memory
- ✅ Use vectorized loads/stores when possible
- ✅ Fuse kernels to reduce memory traffic
- ✅ Profile memory usage regularly

### Don'ts
- ❌ Don't use excessive registers (reduces occupancy)
- ❌ Don't ignore memory alignment
- ❌ Don't create unnecessary temporary arrays
- ❌ Don't use strided memory access patterns
- ❌ Don't forget to synchronize after shared memory writes
- ❌ Don't allocate/deallocate memory in hot paths

## Memory Optimization Checklist

### Before Optimization
- [ ] Profile current memory usage
- [ ] Identify memory bottlenecks
- [ ] Measure memory bandwidth utilization
- [ ] Check for memory access patterns

### During Optimization
- [ ] Implement coalesced access
- [ ] Add shared memory usage
- [ ] Minimize register usage
- [ ] Use vectorized operations
- [ ] Fuse related kernels

### After Optimization
- [ ] Verify correctness
- [ ] Measure performance improvement
- [ ] Check occupancy changes
- [ ] Profile memory bandwidth again

## Common Memory Issues

### Issue: Low Memory Bandwidth
**Symptoms**: GPU utilization low, memory throughput poor
**Solutions**: Improve coalescing, use shared memory, fuse kernels

### Issue: High Register Usage
**Symptoms**: Low occupancy, register spilling warnings
**Solutions**: Reduce temporary variables, use `__launch_bounds__`

### Issue: Bank Conflicts
**Symptoms**: Shared memory access inefficient
**Solutions**: Pad arrays, change access patterns

### Issue: Memory Leaks
**Symptoms**: Increasing memory usage over time
**Solutions**: Match `cudaMalloc` with `cudaFree`, use RAII patterns
