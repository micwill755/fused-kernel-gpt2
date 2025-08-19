# CUDA Programming Basics

## CUDA Execution Model

### Thread Hierarchy
```
Grid (entire kernel launch)
├── Block 0
│   ├── Thread 0, 1, 2, ..., 255
├── Block 1  
│   ├── Thread 0, 1, 2, ..., 255
└── Block N
    ├── Thread 0, 1, 2, ..., 255
```

### Built-in Variables
- `gridDim.x`: Number of blocks in the grid
- `blockDim.x`: Number of threads per block
- `blockIdx.x`: Block index (0 to gridDim.x-1)
- `threadIdx.x`: Thread index within block (0 to blockDim.x-1)

### Global Thread Index
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## Memory Hierarchy

### Memory Types (Speed: Fast → Slow)
1. **Registers**: Per-thread, fastest
2. **Shared Memory**: Per-block, fast, manually managed
3. **Global Memory**: All threads, slow, large capacity
4. **Constant Memory**: Read-only, cached
5. **Texture Memory**: Read-only, cached, optimized for 2D access

### Memory Declaration
```cuda
__global__ void kernel() {
    int reg_var;                    // Register memory
    __shared__ float shared_data[256]; // Shared memory
    // Global memory accessed via pointers
}
```

## Kernel Launch Configuration

### Basic Launch
```cuda
kernel<<<blocks, threads_per_block>>>(args);
```

### With Shared Memory
```cuda
kernel<<<blocks, threads_per_block, shared_mem_size>>>(args);
```

### Common Thread Block Sizes
- **32**: One warp (minimum for efficiency)
- **128**: Good for memory-bound kernels
- **256**: Most common choice
- **512**: Good for compute-bound kernels
- **1024**: Maximum on most GPUs

## Synchronization

### Thread Synchronization
```cuda
__syncthreads(); // All threads in block wait here
```

### Warp-level Operations
```cuda
__syncwarp();           // Synchronize warp
int val = __shfl_down_sync(0xffffffff, value, 1); // Shuffle data
```

## Memory Access Patterns

### Coalesced Access (Good)
```cuda
// Threads access consecutive memory locations
int idx = blockIdx.x * blockDim.x + threadIdx.x;
output[idx] = input[idx];
```

### Strided Access (Bad)
```cuda
// Threads access memory with large strides
int idx = threadIdx.x * stride;
output[idx] = input[idx];
```

## Common Patterns

### Reduction
```cuda
__shared__ float sdata[256];
int tid = threadIdx.x;

// Load data to shared memory
sdata[tid] = input[blockIdx.x * blockDim.x + tid];
__syncthreads();

// Reduce in shared memory
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

### Matrix Multiplication (Tiled)
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

for (int tile = 0; tile < num_tiles; ++tile) {
    // Load tile to shared memory
    As[ty][tx] = A[row * width + tile * TILE_SIZE + tx];
    Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * width + col];
    __syncthreads();
    
    // Compute partial result
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
}
```

## Error Handling

### Check CUDA Errors
```cuda
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
}
```

### Macro for Error Checking
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

## Performance Tips

### Occupancy
- **Goal**: Keep GPU busy with enough threads
- **Tool**: CUDA Occupancy Calculator
- **Rule**: Aim for 50%+ occupancy

### Memory Bandwidth
- **Coalesce memory accesses**
- **Use shared memory for reused data**
- **Minimize global memory transactions**

### Instruction Throughput
- **Avoid divergent branches**
- **Use fast math functions when possible**
- **Minimize register usage**

## Debugging Tools

### Command Line
```bash
cuda-gdb ./program          # Debug CUDA programs
nvprof ./program           # Profile CUDA programs
nsight-compute ./program   # Detailed kernel analysis
```

### In Code
```cuda
printf("Thread %d: value = %f\n", threadIdx.x, value);
```

## Common Mistakes

1. **Not checking array bounds**
2. **Forgetting `__syncthreads()`**
3. **Race conditions in shared memory**
4. **Inefficient memory access patterns**
5. **Too few threads (low occupancy)**
6. **Too many registers (low occupancy)**

## Next Steps

- Practice with simple kernels
- Learn about warp-level primitives
- Study memory optimization techniques
- Explore cooperative groups
- Learn about multi-GPU programming
