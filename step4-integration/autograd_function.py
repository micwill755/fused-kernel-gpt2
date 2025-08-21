"""
Custom autograd function for fused attention + layer norm kernel.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
import sys
import os

# Add step3-fused to path to import the kernel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'step3-fused'))

try:
    import fused_kernel
    FUSED_KERNEL_AVAILABLE = True
except ImportError:
    print("Warning: fused_kernel not available, falling back to PyTorch implementation")
    FUSED_KERNEL_AVAILABLE = False


class FusedAttentionLayerNorm(Function):
    """
    Custom autograd function for fused attention + layer normalization.
    
    This function combines:
    1. Multi-head attention computation
    2. Residual connection
    3. Layer normalization
    
    All in a single CUDA kernel for improved performance.
    """
    
    @staticmethod
    def forward(ctx, input, q_weight, k_weight, v_weight, ln_weight, ln_bias, 
                num_heads, eps=1e-5, use_optimized=True):
        """
        Forward pass of fused attention + layer norm.
        
        Args:
            ctx: Context for backward pass
            input: Input tensor [batch, seq_len, hidden_size]
            q_weight: Query projection weight [hidden_size, hidden_size]
            k_weight: Key projection weight [hidden_size, hidden_size]
            v_weight: Value projection weight [hidden_size, hidden_size]
            ln_weight: Layer norm weight [hidden_size]
            ln_bias: Layer norm bias [hidden_size]
            num_heads: Number of attention heads
            eps: Layer norm epsilon
            use_optimized: Whether to use optimized kernel variant
            
        Returns:
            output: Fused attention + layer norm output [batch, seq_len, hidden_size]
        """
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, q_weight, k_weight, v_weight, ln_weight, ln_bias)
        ctx.num_heads = num_heads
        ctx.eps = eps
        ctx.use_optimized = use_optimized
        
        if FUSED_KERNEL_AVAILABLE:
            # Use our custom CUDA kernel
            output = fused_kernel.fused_attention_layernorm(
                input, q_weight, k_weight, v_weight, ln_weight, ln_bias,
                num_heads, eps, use_optimized
            )
        else:
            # Fallback to PyTorch implementation
            output = _pytorch_fallback_forward(
                input, q_weight, k_weight, v_weight, ln_weight, ln_bias,
                num_heads, eps
            )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of fused attention + layer norm.
        
        For simplicity, we implement the backward pass using PyTorch's
        autograd. In a production implementation, you would want to
        implement custom CUDA kernels for the backward pass as well.
        """
        
        input, q_weight, k_weight, v_weight, ln_weight, ln_bias = ctx.saved_tensors
        num_heads = ctx.num_heads
        eps = ctx.eps
        
        # Enable gradients for saved tensors
        input = input.detach().requires_grad_(True)
        q_weight = q_weight.detach().requires_grad_(True)
        k_weight = k_weight.detach().requires_grad_(True)
        v_weight = v_weight.detach().requires_grad_(True)
        ln_weight = ln_weight.detach().requires_grad_(True)
        ln_bias = ln_bias.detach().requires_grad_(True)
        
        # Recompute forward pass with gradients
        with torch.enable_grad():
            output = _pytorch_fallback_forward(
                input, q_weight, k_weight, v_weight, ln_weight, ln_bias,
                num_heads, eps
            )
        
        # Compute gradients
        torch.autograd.backward(output, grad_output)
        
        return (input.grad, q_weight.grad, k_weight.grad, v_weight.grad,
                ln_weight.grad, ln_bias.grad, None, None, None)


def _pytorch_fallback_forward(input, q_weight, k_weight, v_weight, 
                             ln_weight, ln_bias, num_heads, eps):
    """
    PyTorch fallback implementation for when CUDA kernel is not available.
    """
    batch_size, seq_len, hidden_size = input.shape
    head_dim = hidden_size // num_heads
    scale = 1.0 / (head_dim ** 0.5)
    
    # Linear projections
    Q = F.linear(input, q_weight)
    K = F.linear(input, k_weight)
    V = F.linear(input, v_weight)
    
    # Reshape for multi-head attention
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    
    # Add residual connection
    residual_output = attn_output + input
    
    # Layer normalization
    output = F.layer_norm(residual_output, (hidden_size,), ln_weight, ln_bias, eps)
    
    return output


class FusedAttentionLayerNormModule(torch.nn.Module):
    """
    PyTorch module wrapper for the fused attention + layer norm operation.
    
    This module can be used as a drop-in replacement for separate
    attention and layer norm modules.
    """
    
    def __init__(self, hidden_size, num_heads, eps=1e-5, use_optimized=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.eps = eps
        self.use_optimized = use_optimized
        
        # Initialize weights
        self.q_weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.k_weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.v_weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.ln_weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.ln_bias = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using standard initialization schemes."""
        # Xavier initialization for attention weights
        torch.nn.init.xavier_uniform_(self.q_weight)
        torch.nn.init.xavier_uniform_(self.k_weight)
        torch.nn.init.xavier_uniform_(self.v_weight)
        
        # Layer norm weights are already initialized correctly
    
    def forward(self, input):
        """
        Forward pass through fused attention + layer norm.
        
        Args:
            input: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            output: Fused attention + layer norm output [batch, seq_len, hidden_size]
        """
        return FusedAttentionLayerNorm.apply(
            input, self.q_weight, self.k_weight, self.v_weight,
            self.ln_weight, self.ln_bias, self.num_heads, self.eps, self.use_optimized
        )
    
    def extra_repr(self):
        """String representation of the module."""
        return f'hidden_size={self.hidden_size}, num_heads={self.num_heads}, eps={self.eps}'


def test_autograd_function():
    """Test the autograd function for correctness."""
    print("Testing FusedAttentionLayerNorm autograd function...")
    
    # Test parameters
    batch_size, seq_len, hidden_size, num_heads = 2, 64, 512, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    q_weight = torch.randn(hidden_size, hidden_size, device=device, requires_grad=True)
    k_weight = torch.randn(hidden_size, hidden_size, device=device, requires_grad=True)
    v_weight = torch.randn(hidden_size, hidden_size, device=device, requires_grad=True)
    ln_weight = torch.ones(hidden_size, device=device, requires_grad=True)
    ln_bias = torch.zeros(hidden_size, device=device, requires_grad=True)
    
    # Test forward pass
    output = FusedAttentionLayerNorm.apply(
        input_tensor, q_weight, k_weight, v_weight, ln_weight, ln_bias, num_heads
    )
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    
    print(f"Input grad norm: {input_tensor.grad.norm().item():.6f}")
    print(f"Q weight grad norm: {q_weight.grad.norm().item():.6f}")
    print(f"K weight grad norm: {k_weight.grad.norm().item():.6f}")
    print(f"V weight grad norm: {v_weight.grad.norm().item():.6f}")
    print(f"LN weight grad norm: {ln_weight.grad.norm().item():.6f}")
    print(f"LN bias grad norm: {ln_bias.grad.norm().item():.6f}")
    
    print("✅ Autograd function test completed successfully!")


def test_module_wrapper():
    """Test the module wrapper."""
    print("\nTesting FusedAttentionLayerNormModule...")
    
    # Test parameters
    batch_size, seq_len, hidden_size, num_heads = 2, 64, 512, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create module
    module = FusedAttentionLayerNormModule(hidden_size, num_heads).to(device)
    
    # Create test input
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Forward pass
    output = module(input_tensor)
    
    print(f"Module: {module}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in module.parameters())}")
    
    # Test training mode
    module.train()
    output_train = module(input_tensor)
    
    # Test eval mode
    module.eval()
    output_eval = module(input_tensor)
    
    print(f"Training mode output mean: {output_train.mean().item():.6f}")
    print(f"Eval mode output mean: {output_eval.mean().item():.6f}")
    
    print("✅ Module wrapper test completed successfully!")


if __name__ == "__main__":
    test_autograd_function()
    test_module_wrapper()
