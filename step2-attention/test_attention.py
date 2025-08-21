#!/usr/bin/env python3
"""
Test script for CUDA attention kernel implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import attention_kernel
    print("✅ Successfully imported attention_kernel")
except ImportError as e:
    print(f"❌ Failed to import attention_kernel: {e}")
    print("Please run: python setup.py build_ext --inplace")
    sys.exit(1)

def create_test_tensors(batch_size, seq_len, hidden_size, device='cuda'):
    """Create test tensors for attention computation."""
    torch.manual_seed(42)
    
    Q = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    return Q, K, V

def pytorch_attention_reference(Q, K, V, num_heads):
    """Reference implementation using PyTorch."""
    batch_size, seq_len, hidden_size = Q.shape
    head_dim = hidden_size // num_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    # Reshape for multi-head attention
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    # Reshape back
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    
    return output

def test_correctness():
    """Test correctness of CUDA attention kernel."""
    print("\n🧪 Testing Correctness...")
    
    # Test parameters
    test_cases = [
        (1, 64, 512, 8),    # Small case
        (2, 128, 768, 12),  # Medium case
        (1, 256, 1024, 16), # Large case
    ]
    
    for batch_size, seq_len, hidden_size, num_heads in test_cases:
        print(f"\nTesting: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, heads={num_heads}")
        
        # Create test data
        Q, K, V = create_test_tensors(batch_size, seq_len, hidden_size)
        
        # Compute reference output
        ref_output = pytorch_attention_reference(Q.clone(), K.clone(), V.clone(), num_heads)
        
        # Compute CUDA output (basic kernel)
        try:
            cuda_output_basic = attention_kernel.attention_forward(Q, K, V, num_heads, False)
            
            # Compare results
            max_diff = torch.max(torch.abs(ref_output - cuda_output_basic)).item()
            rel_error = (torch.norm(ref_output - cuda_output_basic) / torch.norm(ref_output)).item()
            
            print(f"  Basic kernel - Max diff: {max_diff:.6f}, Rel error: {rel_error:.6f}")
            
            if max_diff < 1e-3 and rel_error < 1e-3:
                print("  ✅ Basic kernel: PASSED")
            else:
                print("  ❌ Basic kernel: FAILED")
                
        except Exception as e:
            print(f"  ❌ Basic kernel failed: {e}")
        
        # Test optimized kernel
        try:
            cuda_output_opt = attention_kernel.attention_forward(Q, K, V, num_heads, True)
            
            max_diff_opt = torch.max(torch.abs(ref_output - cuda_output_opt)).item()
            rel_error_opt = (torch.norm(ref_output - cuda_output_opt) / torch.norm(ref_output)).item()
            
            print(f"  Optimized kernel - Max diff: {max_diff_opt:.6f}, Rel error: {rel_error_opt:.6f}")
            
            if max_diff_opt < 1e-3 and rel_error_opt < 1e-3:
                print("  ✅ Optimized kernel: PASSED")
            else:
                print("  ❌ Optimized kernel: FAILED")
                
        except Exception as e:
            print(f"  ❌ Optimized kernel failed: {e}")

def test_performance():
    """Test performance of CUDA attention kernel."""
    print("\n⚡ Testing Performance...")
    
    # Performance test parameters
    test_configs = [
        (1, 128, 768, 12),   # GPT-2 small
        (1, 256, 1024, 16),  # GPT-2 medium
        (2, 512, 1536, 24),  # GPT-2 large
    ]
    
    num_iterations = 50
    
    for batch_size, seq_len, hidden_size, num_heads in test_configs:
        print(f"\nBenchmarking: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, heads={num_heads}")
        
        # Create test data
        Q, K, V = create_test_tensors(batch_size, seq_len, hidden_size)
        
        try:
            # Run benchmark
            times = attention_kernel.benchmark_attention(Q, K, V, num_heads, num_iterations)
            basic_time, optimized_time, pytorch_time = times
            
            print(f"  PyTorch reference: {pytorch_time:.3f} ms")
            print(f"  Basic CUDA kernel: {basic_time:.3f} ms ({pytorch_time/basic_time:.2f}x speedup)")
            print(f"  Optimized kernel:  {optimized_time:.3f} ms ({pytorch_time/optimized_time:.2f}x speedup)")
            
            # Memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"  Peak memory usage: {memory_used:.1f} MB")
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency and limits."""
    print("\n💾 Testing Memory Efficiency...")
    
    # Test with increasing sequence lengths
    batch_size = 1
    hidden_size = 768
    num_heads = 12
    
    for seq_len in [128, 256, 512, 1024]:
        print(f"\nTesting seq_len={seq_len}")
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            Q, K, V = create_test_tensors(batch_size, seq_len, hidden_size)
            
            # Run attention
            output = attention_kernel.attention_forward(Q, K, V, num_heads, True)
            
            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            theoretical_memory = (3 * batch_size * seq_len * hidden_size * 4 + 
                                 batch_size * num_heads * seq_len * seq_len * 4) / 1024**2
            
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Theoretical minimum: {theoretical_memory:.1f} MB")
            print(f"  Efficiency: {theoretical_memory/memory_used*100:.1f}%")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Out of memory at seq_len={seq_len}")
                break
            else:
                raise e

def main():
    """Main test function."""
    print("🚀 CUDA Attention Kernel Test Suite")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return
    
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run tests
    try:
        test_correctness()
        test_performance()
        test_memory_efficiency()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
