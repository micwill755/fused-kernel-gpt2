# Step 2: Attention Kernel

## Goal
Implement a CUDA kernel for the attention mechanism used in GPT-2.

## What You'll Learn
- Matrix multiplication in CUDA
- Softmax computation
- Shared memory optimization
- Multi-dimensional tensor indexing
- Attention mechanism implementation

## Attention Formula
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where:
- Q: Query matrix [batch, seq_len, hidden_size]
- K: Key matrix [batch, seq_len, hidden_size]  
- V: Value matrix [batch, seq_len, hidden_size]
- d_k: Dimension of key vectors (hidden_size / num_heads)

## Implementation Steps

1. **Matrix Multiplication**: Q @ K^T
2. **Scaling**: Divide by sqrt(d_k)
3. **Softmax**: Apply softmax along sequence dimension
4. **Final Multiplication**: Result @ V

## Files to Create

- `attention_kernel.cu` - CUDA kernel implementation
- `attention_host.cpp` - Host wrapper function
- `setup.py` - Build configuration
- `test_attention.py` - Test script

## Key Challenges

### Memory Management
- Efficient shared memory usage for matrix tiles
- Handling variable sequence lengths
- Managing multiple attention heads

### Numerical Stability
- Avoiding overflow in softmax computation
- Maintaining precision with FP16

### Performance Optimization
- Memory coalescing
- Reducing global memory accesses
- Optimal thread block dimensions

## Expected Performance
Target: 2-3x speedup over naive PyTorch implementation for specific batch sizes and sequence lengths.

## Next Steps
After completing this step, you'll move to Step 3 where you'll fuse this attention kernel with layer normalization and residual connections.

## Hints

1. Start with a simple implementation that works
2. Use shared memory for matrix tiles
3. Handle softmax carefully (subtract max for stability)
4. Test with small matrices first
5. Profile to identify bottlenecks

## Resources
- CUDA Programming Guide: Matrix Multiplication
- Flash Attention paper for advanced optimizations
- cuBLAS documentation for reference implementations
