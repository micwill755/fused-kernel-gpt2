#!/usr/bin/env python3
"""
Test script for Step 1: Simple CUDA Kernel
Run this after building the extension to verify it works correctly.
"""

import torch
import time
import simple_kernel

def test_correctness():
    """Test that our CUDA kernel produces correct results"""
    print("🧪 Testing kernel correctness...")
    
    # Create test tensors
    size = (1000, 1000)  # 1M elements
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Compute using our CUDA kernel
    result_cuda = simple_kernel.simple_add(a, b)
    
    # Compute using PyTorch (reference)
    result_pytorch = a + b
    
    # Check if results match
    if torch.allclose(result_cuda, result_pytorch, rtol=1e-5):
        print("✅ Correctness test PASSED!")
        return True
    else:
        print("❌ Correctness test FAILED!")
        print(f"Max difference: {torch.max(torch.abs(result_cuda - result_pytorch))}")
        return False

def benchmark_performance():
    """Compare performance of our kernel vs PyTorch"""
    print("\n⚡ Benchmarking performance...")
    
    # Test different sizes
    sizes = [(100, 100), (1000, 1000), (5000, 5000)]
    
    for size in sizes:
        print(f"\nTesting size: {size[0]}x{size[1]} = {size[0]*size[1]:,} elements")
        
        # Create test tensors
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Warm up GPU
        for _ in range(10):
            _ = simple_kernel.simple_add(a, b)
            _ = a + b
        torch.cuda.synchronize()
        
        # Benchmark our CUDA kernel
        start_time = time.time()
        for _ in range(100):
            result_cuda = simple_kernel.simple_add(a, b)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / 100
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            result_pytorch = a + b
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 100
        
        # Calculate speedup
        speedup = pytorch_time / cuda_time
        
        print(f"  Our CUDA kernel: {cuda_time*1000:.3f} ms")
        print(f"  PyTorch:         {pytorch_time*1000:.3f} ms")
        print(f"  Speedup:         {speedup:.2f}x")

def main():
    """Main test function"""
    print("🚀 Testing Simple CUDA Kernel (Step 1)")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please check your installation.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Run tests
    if test_correctness():
        benchmark_performance()
        print("\n🎉 All tests completed successfully!")
        print("\n📚 Next step: Go to step2-attention/ to implement attention kernel")
    else:
        print("\n❌ Tests failed. Please check your implementation.")

if __name__ == "__main__":
    main()
