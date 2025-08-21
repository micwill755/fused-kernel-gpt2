"""
Modified GPT-2 model with fused attention + layer norm kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention
import math
import warnings

from autograd_function import FusedAttentionLayerNormModule


class OptimizedGPT2Attention(nn.Module):
    """
    Optimized GPT-2 attention using fused kernel.
    
    This replaces the standard GPT-2 attention + layer norm with our
    fused implementation for better performance.
    """
    
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Use fused attention + layer norm
        self.fused_attn_ln = FusedAttentionLayerNormModule(
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            eps=config.layer_norm_epsilon,
            use_optimized=True
        )
        
        # Output projection (separate from fused kernel)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        """Prune attention heads (not implemented for fused kernel)."""
        if len(heads) == 0:
            return
        warnings.warn("Head pruning not implemented for fused attention kernel")
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Standard attention computation (fallback for cross-attention).
        """
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split hidden states into multiple attention heads."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge attention heads back into hidden states."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            # Cross-attention case - use standard implementation
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn`, `c_attn`, `c_proj` have to be defined."
                )
            
            # This would require implementing cross-attention in the fused kernel
            # For now, fall back to standard implementation
            raise NotImplementedError("Cross-attention not implemented for fused kernel")
        
        # Self-attention case - use fused kernel
        if layer_past is not None or use_cache or output_attentions:
            # For cases requiring past key/values or attention weights output,
            # fall back to standard implementation
            warnings.warn("Falling back to standard attention for advanced features")
            return self._standard_attention_forward(
                hidden_states, layer_past, attention_mask, head_mask,
                use_cache, output_attentions
            )
        
        # Apply causal mask to attention mask if needed
        if attention_mask is not None:
            # Convert attention mask to the format expected by our kernel
            # This is a simplified version - production code would need more careful handling
            batch_size, seq_len = hidden_states.shape[:2]
            causal_mask = self.bias[:, :, :seq_len, :seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            combined_mask = causal_mask & attention_mask.bool()
            
            # For simplicity, we'll ignore the attention mask in the fused kernel
            # Production implementation would need to handle this properly
            if not combined_mask.all():
                warnings.warn("Attention mask not fully supported in fused kernel")
        
        # Use fused attention + layer norm
        attn_output = self.fused_attn_ln(hidden_states)
        
        # Apply output projection and residual dropout
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (None,)  # No attention weights available from fused kernel
        if use_cache:
            outputs += (None,)  # No past key/values from fused kernel
        
        return outputs
    
    def _standard_attention_forward(self, hidden_states, layer_past, attention_mask, 
                                  head_mask, use_cache, output_attentions):
        """Fallback to standard attention implementation."""
        # This is a simplified fallback - in practice, you'd want to implement
        # the full GPT-2 attention logic here
        raise NotImplementedError("Standard attention fallback not implemented")


class OptimizedGPT2Block(GPT2Block):
    """
    Optimized GPT-2 block using fused attention kernel.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        # Replace standard attention with optimized version
        self.attn = OptimizedGPT2Attention(config, layer_idx=layer_idx)
        
        # Note: We've fused the layer norm into the attention, so we remove ln_1
        # The fused kernel handles: attention + residual + layer_norm
        # We still need ln_2 for the MLP block
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = self.mlp  # Keep existing MLP
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # Attention block (includes residual connection and layer norm)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # MLP block
        residual = attn_output
        hidden_states = self.ln_2(attn_output)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs


class OptimizedGPT2Model(GPT2Model):
    """
    Optimized GPT-2 model using fused kernels.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Replace transformer blocks with optimized versions
        self.h = nn.ModuleList([OptimizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
        # Initialize weights
        self.post_init()
    
    @classmethod
    def from_pretrained_gpt2(cls, model_name_or_path, **kwargs):
        """
        Load a pre-trained GPT-2 model and convert it to use fused kernels.
        """
        # Load standard GPT-2 model
        standard_model = GPT2Model.from_pretrained(model_name_or_path, **kwargs)
        
        # Create optimized model with same config
        optimized_model = cls(standard_model.config)
        
        # Copy weights from standard model
        optimized_model.load_state_dict(standard_model.state_dict(), strict=False)
        
        # Copy attention weights to fused modules
        for i, (std_block, opt_block) in enumerate(zip(standard_model.h, optimized_model.h)):
            # Copy attention weights
            std_attn = std_block.attn
            opt_attn = opt_block.attn.fused_attn_ln
            
            # Extract Q, K, V weights from combined c_attn weight
            c_attn_weight = std_attn.c_attn.weight
            hidden_size = c_attn_weight.shape[1]
            
            # Split combined QKV weight
            q_weight, k_weight, v_weight = c_attn_weight.chunk(3, dim=0)
            
            # Copy to fused module
            opt_attn.q_weight.data = q_weight
            opt_attn.k_weight.data = k_weight
            opt_attn.v_weight.data = v_weight
            
            # Copy layer norm weights
            opt_attn.ln_weight.data = std_block.ln_1.weight.data
            opt_attn.ln_bias.data = std_block.ln_1.bias.data
            
            # Copy output projection
            opt_block.attn.c_proj.weight.data = std_attn.c_proj.weight.data
            opt_block.attn.c_proj.bias.data = std_attn.c_proj.bias.data
        
        return optimized_model


class OptimizedGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Optimized GPT-2 language model using fused kernels.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Replace transformer with optimized version
        self.transformer = OptimizedGPT2Model(config)
        
        # Initialize weights
        self.post_init()
    
    @classmethod
    def from_pretrained_gpt2(cls, model_name_or_path, **kwargs):
        """
        Load a pre-trained GPT-2 language model and convert it to use fused kernels.
        """
        # Load standard GPT-2 model
        standard_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, **kwargs)
        
        # Create optimized model with same config
        optimized_model = cls(standard_model.config)
        
        # Copy transformer weights
        optimized_model.transformer = OptimizedGPT2Model.from_pretrained_gpt2(model_name_or_path, **kwargs)
        
        # Copy language modeling head
        optimized_model.lm_head.weight.data = standard_model.lm_head.weight.data
        
        return optimized_model


def test_optimized_model():
    """Test the optimized GPT-2 model."""
    print("Testing OptimizedGPT2Model...")
    
    # Create a small config for testing
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,  # Small for testing
        n_head=12,
        n_inner=3072,
    )
    
    # Create optimized model
    model = OptimizedGPT2Model(config)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test language model
    lm_model = OptimizedGPT2LMHeadModel(config)
    if torch.cuda.is_available():
        lm_model = lm_model.cuda()
    
    with torch.no_grad():
        lm_outputs = lm_model(input_ids)
    
    print(f"LM output shape: {lm_outputs.logits.shape}")
    print("✅ Optimized model test completed successfully!")


if __name__ == "__main__":
    test_optimized_model()
