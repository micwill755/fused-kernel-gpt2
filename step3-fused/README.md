# Step 3: Fused Kernel

## Goal
Implement a fused CUDA kernel that combines attention computation, layer normalization, and residual connection in a single kernel launch.

## What You'll Learn
- Kernel fusion techniques
- Memory bandwidth optimization
- Register and shared memory management
- Complex kernel orchestration
- Performance analysis of fused operations

## Fusion Pattern
```
Input → Attention → Add Residual → Layer Norm → Output
```

Instead of three separate kernel launches:
1. `attention_output = attention(input)`
2. `residual_output = attention_output + input`
3. `final_output = layer_norm(residual_output)`

We perform all operations in a single kernel:
```cuda
__global__ void fused_attention_layernorm_kernel(
    input, output, attention_weights, layer_norm_weights, layer_norm_bias
)
```

## Key Benefits
- **Reduced Memory Bandwidth**: Intermediate results stay in registers/shared memory
- **Fewer Kernel Launches**: Reduced CPU-GPU synchronization overhead
- **Better Cache Utilization**: Data reuse within the same kernel
- **Lower Latency**: Eliminates intermediate global memory writes

## Implementation Strategy

### Phase 1: Attention Computation
- Load Q, K, V from global memory
- Compute attention scores in shared memory
- Apply softmax with numerical stability
- Compute attention output

### Phase 2: Residual Connection
- Add original input to attention output
- Keep result in registers (avoid global memory write)

### Phase 3: Layer Normalization
- Compute mean and variance across hidden dimension
- Apply normalization with learned parameters
- Write final result to global memory

## Memory Management

### Shared Memory Layout
```
[Q_tile][K_tile][V_tile][Attention_Scores][LayerNorm_Stats]
```

### Register Usage
- Minimize register pressure for high occupancy
- Reuse registers between phases
- Careful scheduling of operations

## Files to Create

- `fused_kernel.cu` - Main fused kernel implementation
- `fused_host.cpp` - Host wrapper functions
- `pybind.cpp` - Python bindings
- `setup.py` - Build configuration
- `test_fused.py` - Comprehensive test suite
- `benchmark.py` - Performance comparison

## Performance Targets

- **Memory Bandwidth**: 50-70% reduction vs separate kernels
- **Latency**: 30-50% improvement for typical GPT-2 sizes
- **Throughput**: 2-3x improvement for batch processing

## Advanced Optimizations

### Warp-level Primitives
- Use `__shfl_*` for efficient reductions
- Cooperative groups for flexible synchronization

### Memory Coalescing
- Ensure all global memory accesses are coalesced
- Use vectorized loads/stores where possible

### Occupancy Optimization
- Balance shared memory usage vs occupancy
- Minimize register usage per thread

## Testing Strategy

### Correctness Tests
1. Compare against separate kernel implementations
2. Test with various input sizes and configurations
3. Verify numerical stability with extreme values

### Performance Tests
1. Measure memory bandwidth utilization
2. Compare against PyTorch implementations
3. Profile with different batch sizes and sequence lengths

## Common Challenges

### Synchronization
- Careful placement of `__syncthreads()`
- Avoiding deadlocks in conditional code

### Memory Pressure
- Balancing shared memory usage
- Managing register spilling

### Numerical Stability
- Maintaining precision through multiple operations
- Handling edge cases in layer normalization

## Next Steps
After completing this step, you'll integrate the fused kernel into a real GPT-2 model in Step 4.

## Debugging Tips

1. Start with a simple version that works
2. Add one optimization at a time
3. Use `printf` debugging for kernel issues
4. Profile with `nsight-compute` for detailed analysis
5. Compare intermediate results with reference implementations
