# Step 2: CUDA Attention Kernel Implementation

This step implements a transformer attention mechanism in pure CUDA C++, focusing on the core Q×K^T, softmax, and attention×V operations for GPT-2. This is the foundation before moving to the fused kernel in Step 3.

## What This Step Does

**Single Operation Focus**: Implements just the attention computation:
```
Input → Attention(Q, K, V) → Output
```

This step does **NOT** include:
- Residual connections
- Layer normalization  
- Other transformer operations

Those will be fused together in Step 3 for performance optimization.

## Files

- `attention_kernel.cu` - CUDA kernel implementation with optimized attention computation
- `main.cpp` - C++ test harness with performance benchmarking
- `attention_host.cpp` - Host-side helper functions
- `test_attention.py` - Python validation script (optional, for comparison with PyTorch)

## Build and Run

### Option 1: Simple nvcc command
```bash
nvcc -O3 attention_kernel.cu main.cpp attention_host.cpp -o attention_test -lcublas
```

### Option 2: Using Makefile
```bash
make                    # Basic build
make multi-arch         # Build with support for multiple GPU architectures
make run               # Build and run
make clean             # Clean build files
```

### Option 3: Full nvcc command (all optimizations)
```bash
nvcc -O3 --use_fast_math -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86 -std=c++14 -lcublas attention_kernel.cu main.cpp attention_host.cpp -o attention_test
```

### Run the test:
```bash
./attention_test
```

## Build Flags Explained

- `-O3` - Maximum compiler optimization
- `--use_fast_math` - Use faster (but less precise) math functions
- `-arch=sm_XX` - Target specific GPU architectures
- `-std=c++14` - Use C++14 standard
- `-lcublas` - Link with cuBLAS library for matrix operations

## What You'll Learn in This Step

1. **Multi-Head Attention Implementation**: 
   - Query, Key, Value matrix operations
   - Scaled dot-product attention
   - Multi-head parallel computation

2. **CUDA Optimization Techniques**:
   - Shared memory usage for tile-based computation
   - Coalesced memory access patterns
   - Warp-level primitives for efficient reductions
   - Softmax implementation with numerical stability

3. **Memory Management**:
   - Efficient Q, K, V matrix loading
   - Attention score computation and storage
   - Output tensor writing

## Key Optimizations

- **Tiled Computation**: Uses shared memory to load tiles of Q, K, V matrices
- **Softmax Fusion**: Computes softmax inline without storing intermediate scores
- **Memory Coalescing**: Ensures efficient memory access patterns
- **Warp Shuffles**: Uses warp-level operations for fast reductions

## Expected Output

The program will run attention computation tests with different configurations and report:
- Correctness validation against reference implementation
- Performance metrics (GFLOPS, memory bandwidth)
- Kernel execution times
- Comparison with cuBLAS baseline

## Architecture Support

The multi-arch build targets multiple GPU architectures:
- sm_70 (V100)
- sm_75 (T4, RTX 20xx)
- sm_80 (A100, RTX 30xx)
- sm_86 (RTX 40xx)

## Next Step: Kernel Fusion

After mastering attention computation in this step, **Step 3** will teach you to fuse multiple operations:

```
Step 2: Input → Attention → Output
Step 3: Input → [Attention + Residual + LayerNorm] → Output (all in one kernel!)
```

Step 3 achieves:
- 50-70% memory bandwidth reduction
- 30-50% latency improvement
- Single kernel launch instead of multiple separate operations

## Learning Progression

- **Step 1**: Basic CUDA kernels
- **Step 2** (Current): Attention mechanism only
- **Step 3**: Fused attention + residual + layer norm
- **Step 4**: Complete GPT-2 model integration
