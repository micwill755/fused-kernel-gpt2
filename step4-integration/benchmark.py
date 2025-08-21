#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for fused CUDA kernels in GPT-2.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import psutil
import gc
from contextlib import contextmanager
import warnings

from gpt2_model import OptimizedGPT2Model, OptimizedGPT2LMHeadModel
from autograd_function import FusedAttentionLayerNormModule

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield lambda: start.elapsed_time(end)
        end.record()
        torch.cuda.synchronize()
    else:
        start_time = time.time()
        yield lambda: (time.time() - start_time) * 1000


class GPT2Benchmark:
    """Comprehensive benchmarking suite for GPT-2 models."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = []
        
        # Model configurations to test
        self.configs = {
            'gpt2-small': GPT2Config(
                vocab_size=50257, n_positions=1024, n_embd=768, 
                n_layer=12, n_head=12, n_inner=3072
            ),
            'gpt2-medium': GPT2Config(
                vocab_size=50257, n_positions=1024, n_embd=1024, 
                n_layer=24, n_head=16, n_inner=4096
            ),
            'gpt2-large': GPT2Config(
                vocab_size=50257, n_positions=1024, n_embd=1280, 
                n_layer=36, n_head=20, n_inner=5120
            ),
        }
        
        # Test configurations
        self.test_configs = [
            {'batch_size': 1, 'seq_len': 128},
            {'batch_size': 1, 'seq_len': 256},
            {'batch_size': 1, 'seq_len': 512},
            {'batch_size': 1, 'seq_len': 1024},
            {'batch_size': 4, 'seq_len': 128},
            {'batch_size': 4, 'seq_len': 256},
            {'batch_size': 8, 'seq_len': 128},
            {'batch_size': 16, 'seq_len': 64},
        ]
    
    def benchmark_single_layer(self, hidden_size, num_heads, batch_size, seq_len, num_iterations=100):
        """Benchmark a single fused attention layer."""
        print(f"Benchmarking single layer: hidden={hidden_size}, heads={num_heads}, "
              f"batch={batch_size}, seq={seq_len}")
        
        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        
        # Standard PyTorch implementation
        standard_attn = torch.nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        ).to(self.device)
        standard_ln = torch.nn.LayerNorm(hidden_size).to(self.device)
        
        # Fused implementation
        fused_layer = FusedAttentionLayerNormModule(
            hidden_size, num_heads
        ).to(self.device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                # Standard
                attn_out, _ = standard_attn(input_tensor, input_tensor, input_tensor)
                std_out = standard_ln(attn_out + input_tensor)
                
                # Fused
                fused_out = fused_layer(input_tensor)
        
        torch.cuda.synchronize()
        
        # Benchmark standard implementation
        times_standard = []
        for _ in range(num_iterations):
            with cuda_timer() as timer:
                with torch.no_grad():
                    attn_out, _ = standard_attn(input_tensor, input_tensor, input_tensor)
                    std_out = standard_ln(attn_out + input_tensor)
            times_standard.append(timer())
        
        # Benchmark fused implementation
        times_fused = []
        for _ in range(num_iterations):
            with cuda_timer() as timer:
                with torch.no_grad():
                    fused_out = fused_layer(input_tensor)
            times_fused.append(timer())
        
        avg_standard = np.mean(times_standard)
        avg_fused = np.mean(times_fused)
        speedup = avg_standard / avg_fused
        
        return {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'standard_time': avg_standard,
            'fused_time': avg_fused,
            'speedup': speedup,
            'memory_saved': self._estimate_memory_savings(batch_size, seq_len, hidden_size)
        }
    
    def benchmark_full_model(self, model_name, batch_size, seq_len, num_iterations=20):
        """Benchmark full GPT-2 model."""
        print(f"Benchmarking {model_name}: batch={batch_size}, seq={seq_len}")
        
        config = self.configs[model_name]
        
        # Create models
        standard_model = GPT2Model(config).to(self.device)
        optimized_model = OptimizedGPT2Model(config).to(self.device)
        
        # Create test input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                _ = standard_model(input_ids)
                _ = optimized_model(input_ids)
        
        torch.cuda.synchronize()
        
        # Benchmark standard model
        times_standard = []
        memory_standard = []
        
        for _ in range(num_iterations):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with cuda_timer() as timer:
                with torch.no_grad():
                    _ = standard_model(input_ids)
            
            times_standard.append(timer())
            memory_standard.append(torch.cuda.max_memory_allocated())
        
        # Benchmark optimized model
        times_optimized = []
        memory_optimized = []
        
        for _ in range(num_iterations):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with cuda_timer() as timer:
                with torch.no_grad():
                    _ = optimized_model(input_ids)
            
            times_optimized.append(timer())
            memory_optimized.append(torch.cuda.max_memory_allocated())
        
        avg_standard = np.mean(times_standard)
        avg_optimized = np.mean(times_optimized)
        avg_mem_standard = np.mean(memory_standard)
        avg_mem_optimized = np.mean(memory_optimized)
        
        return {
            'model_name': model_name,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'standard_time': avg_standard,
            'optimized_time': avg_optimized,
            'speedup': avg_standard / avg_optimized,
            'standard_memory': avg_mem_standard / 1024**2,  # MB
            'optimized_memory': avg_mem_optimized / 1024**2,  # MB
            'memory_savings': (avg_mem_standard - avg_mem_optimized) / avg_mem_standard * 100
        }
    
    def benchmark_throughput(self, model_name, max_batch_size=32, seq_len=128):
        """Benchmark throughput scaling with batch size."""
        print(f"Benchmarking throughput for {model_name}")
        
        config = self.configs[model_name]
        standard_model = GPT2Model(config).to(self.device)
        optimized_model = OptimizedGPT2Model(config).to(self.device)
        
        batch_sizes = [1, 2, 4, 8, 16]
        if max_batch_size > 16:
            batch_sizes.extend([24, 32])
        
        results = []
        
        for batch_size in batch_sizes:
            try:
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)
                
                # Test standard model
                torch.cuda.empty_cache()
                times = []
                for _ in range(10):
                    with cuda_timer() as timer:
                        with torch.no_grad():
                            _ = standard_model(input_ids)
                    times.append(timer())
                
                standard_time = np.mean(times)
                standard_throughput = (batch_size * seq_len) / (standard_time / 1000)  # tokens/sec
                
                # Test optimized model
                torch.cuda.empty_cache()
                times = []
                for _ in range(10):
                    with cuda_timer() as timer:
                        with torch.no_grad():
                            _ = optimized_model(input_ids)
                    times.append(timer())
                
                optimized_time = np.mean(times)
                optimized_throughput = (batch_size * seq_len) / (optimized_time / 1000)  # tokens/sec
                
                results.append({
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'standard_throughput': standard_throughput,
                    'optimized_throughput': optimized_throughput,
                    'throughput_improvement': optimized_throughput / standard_throughput
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at batch_size={batch_size}")
                    break
                else:
                    raise e
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite."""
        print("🚀 Starting Comprehensive GPT-2 Fused Kernel Benchmark")
        print("=" * 60)
        
        all_results = []
        
        # 1. Single layer benchmarks
        print("\n📊 Single Layer Benchmarks")
        print("-" * 30)
        
        layer_configs = [
            (768, 12),   # GPT-2 small
            (1024, 16),  # GPT-2 medium
            (1280, 20),  # GPT-2 large
        ]
        
        for hidden_size, num_heads in layer_configs:
            for test_config in self.test_configs[:4]:  # Subset for single layer
                result = self.benchmark_single_layer(
                    hidden_size, num_heads, 
                    test_config['batch_size'], test_config['seq_len']
                )
                all_results.append(result)
                print(f"  {hidden_size}d, {num_heads}h, {test_config['batch_size']}x{test_config['seq_len']}: "
                      f"{result['speedup']:.2f}x speedup")
        
        # 2. Full model benchmarks
        print("\n🏗️ Full Model Benchmarks")
        print("-" * 30)
        
        model_results = []
        for model_name in ['gpt2-small']:  # Start with small model
            for test_config in self.test_configs[:6]:  # Subset for full model
                try:
                    result = self.benchmark_full_model(
                        model_name, test_config['batch_size'], test_config['seq_len']
                    )
                    model_results.append(result)
                    print(f"  {model_name} {test_config['batch_size']}x{test_config['seq_len']}: "
                          f"{result['speedup']:.2f}x speedup, {result['memory_savings']:.1f}% memory saved")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  {model_name} {test_config['batch_size']}x{test_config['seq_len']}: OOM")
                    else:
                        raise e
        
        # 3. Throughput benchmarks
        print("\n⚡ Throughput Benchmarks")
        print("-" * 30)
        
        throughput_results = []
        for model_name in ['gpt2-small']:
            results = self.benchmark_throughput(model_name)
            throughput_results.extend(results)
            
            for result in results:
                print(f"  {model_name} batch={result['batch_size']}: "
                      f"{result['throughput_improvement']:.2f}x throughput improvement")
        
        # 4. Generate summary report
        self._generate_report(all_results, model_results, throughput_results)
        
        return all_results, model_results, throughput_results
    
    def _estimate_memory_savings(self, batch_size, seq_len, hidden_size):
        """Estimate memory savings from kernel fusion."""
        # Intermediate tensors saved by fusion
        attention_scores = batch_size * seq_len * seq_len * 4  # bytes
        attention_output = batch_size * seq_len * hidden_size * 4
        residual_temp = batch_size * seq_len * hidden_size * 4
        
        total_saved = attention_scores + attention_output + residual_temp
        return total_saved / 1024**2  # MB
    
    def _generate_report(self, layer_results, model_results, throughput_results):
        """Generate comprehensive benchmark report."""
        print("\n📈 Benchmark Summary Report")
        print("=" * 50)
        
        if layer_results:
            layer_df = pd.DataFrame(layer_results)
            print(f"\n🔧 Single Layer Performance:")
            print(f"  Average speedup: {layer_df['speedup'].mean():.2f}x")
            print(f"  Best speedup: {layer_df['speedup'].max():.2f}x")
            print(f"  Average memory saved: {layer_df['memory_saved'].mean():.1f} MB")
        
        if model_results:
            model_df = pd.DataFrame(model_results)
            print(f"\n🏗️ Full Model Performance:")
            print(f"  Average speedup: {model_df['speedup'].mean():.2f}x")
            print(f"  Best speedup: {model_df['speedup'].max():.2f}x")
            print(f"  Average memory savings: {model_df['memory_savings'].mean():.1f}%")
        
        if throughput_results:
            throughput_df = pd.DataFrame(throughput_results)
            print(f"\n⚡ Throughput Performance:")
            print(f"  Average throughput improvement: {throughput_df['throughput_improvement'].mean():.2f}x")
            print(f"  Best throughput improvement: {throughput_df['throughput_improvement'].max():.2f}x")
        
        # System info
        print(f"\n💻 System Information:")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CPU: {psutil.cpu_count()} cores")
        print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    def plot_results(self, layer_results, model_results, throughput_results):
        """Generate visualization plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Layer speedup vs sequence length
        if layer_results:
            layer_df = pd.DataFrame(layer_results)
            for hidden_size in layer_df['hidden_size'].unique():
                subset = layer_df[layer_df['hidden_size'] == hidden_size]
                axes[0, 0].plot(subset['seq_len'], subset['speedup'], 
                               marker='o', label=f'{hidden_size}d')
            
            axes[0, 0].set_xlabel('Sequence Length')
            axes[0, 0].set_ylabel('Speedup (x)')
            axes[0, 0].set_title('Single Layer Speedup vs Sequence Length')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Model speedup vs batch size
        if model_results:
            model_df = pd.DataFrame(model_results)
            for seq_len in model_df['seq_len'].unique():
                subset = model_df[model_df['seq_len'] == seq_len]
                axes[0, 1].plot(subset['batch_size'], subset['speedup'], 
                               marker='s', label=f'seq_len={seq_len}')
            
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Speedup (x)')
            axes[0, 1].set_title('Full Model Speedup vs Batch Size')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Memory savings
        if model_results:
            model_df = pd.DataFrame(model_results)
            axes[1, 0].bar(range(len(model_df)), model_df['memory_savings'])
            axes[1, 0].set_xlabel('Test Configuration')
            axes[1, 0].set_ylabel('Memory Savings (%)')
            axes[1, 0].set_title('Memory Savings by Configuration')
            axes[1, 0].grid(True)
        
        # Plot 4: Throughput improvement
        if throughput_results:
            throughput_df = pd.DataFrame(throughput_results)
            axes[1, 1].plot(throughput_df['batch_size'], throughput_df['throughput_improvement'], 
                           marker='d', color='green')
            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Throughput Improvement (x)')
            axes[1, 1].set_title('Throughput Improvement vs Batch Size')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main benchmarking function."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Benchmarks require GPU.")
        return
    
    benchmark = GPT2Benchmark()
    
    try:
        layer_results, model_results, throughput_results = benchmark.run_comprehensive_benchmark()
        
        # Generate plots
        benchmark.plot_results(layer_results, model_results, throughput_results)
        
        print("\n✅ Benchmark completed successfully!")
        print("📊 Results saved to benchmark_results.png")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
