#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define TILE_SIZE 16

// Utility functions
__device__ __forceinline__ float safe_divide(float a, float b) {
    return b == 0.0f ? 0.0f : a / b;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction for sums
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction among warp leaders
    if (warp_id == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Fused attention + residual + layer norm kernel
__global__ void fused_attention_layernorm_kernel(
    const float* input,           // [batch, seq_len, hidden_size]
    const float* Q_weight,        // [hidden_size, hidden_size]
    const float* K_weight,        // [hidden_size, hidden_size]
    const float* V_weight,        // [hidden_size, hidden_size]
    const float* ln_weight,       // [hidden_size]
    const float* ln_bias,         // [hidden_size]
    float* output,                // [batch, seq_len, hidden_size]
    float* attention_scores,      // [batch, num_heads, seq_len, seq_len] - temporary
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    float eps = 1e-5f
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= num_heads) return;
    
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / sqrtf(head_dim);
    
    // Shared memory layout
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + head_dim;
    float* shared_v = shared_mem + 2 * head_dim;
    float* shared_scores = shared_mem + 3 * head_dim;
    float* shared_attn_out = shared_mem + 3 * head_dim + seq_len;
    float* reduction_mem = shared_mem + 3 * head_dim + seq_len + head_dim;
    
    // Input offset for current sequence position
    int input_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Phase 1: Compute Q, K, V projections for current head
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    
    if (tid < head_dim) {
        // Load input
        float input_val = input[input_offset + head_idx * head_dim + tid];
        
        // Compute Q projection (simplified - in practice, use proper matrix multiplication)
        for (int i = 0; i < hidden_size; i++) {
            float weight_q = Q_weight[(head_idx * head_dim + tid) * hidden_size + i];
            q_val += input_val * weight_q;
        }
        
        // Similarly for K and V (simplified)
        for (int i = 0; i < hidden_size; i++) {
            float weight_k = K_weight[(head_idx * head_dim + tid) * hidden_size + i];
            float weight_v = V_weight[(head_idx * head_dim + tid) * hidden_size + i];
            k_val += input_val * weight_k;
            v_val += input_val * weight_v;
        }
        
        shared_q[tid] = q_val;
        shared_k[tid] = k_val;
        shared_v[tid] = v_val;
    }
    __syncthreads();
    
    // Phase 2: Compute attention scores
    for (int k_seq = tid; k_seq < seq_len; k_seq += blockDim.x) {
        float score = 0.0f;
        
        // Load K for position k_seq
        for (int d = 0; d < head_dim; d++) {
            int k_input_offset = batch_idx * seq_len * hidden_size + k_seq * hidden_size + head_idx * head_dim + d;
            float k_val_pos = 0.0f;
            
            // Compute K projection for position k_seq
            for (int i = 0; i < hidden_size; i++) {
                float input_k = input[batch_idx * seq_len * hidden_size + k_seq * hidden_size + i];
                float weight_k = K_weight[(head_idx * head_dim + d) * hidden_size + i];
                k_val_pos += input_k * weight_k;
            }
            
            score += shared_q[d] * k_val_pos;
        }
        
        shared_scores[k_seq] = score * scale;
    }
    __syncthreads();
    
    // Phase 3: Softmax
    if (tid == 0) {
        // Find max for numerical stability
        float max_score = -INFINITY;
        for (int i = 0; i < seq_len; i++) {
            max_score = fmaxf(max_score, shared_scores[i]);
        }
        
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            shared_scores[i] = expf(shared_scores[i] - max_score);
            sum_exp += shared_scores[i];
        }
        
        // Normalize
        for (int i = 0; i < seq_len; i++) {
            shared_scores[i] = safe_divide(shared_scores[i], sum_exp);
        }
    }
    __syncthreads();
    
    // Phase 4: Compute attention output
    if (tid < head_dim) {
        float attn_result = 0.0f;
        
        for (int v_seq = 0; v_seq < seq_len; v_seq++) {
            float weight = shared_scores[v_seq];
            
            // Compute V projection for position v_seq
            float v_val_pos = 0.0f;
            for (int i = 0; i < hidden_size; i++) {
                float input_v = input[batch_idx * seq_len * hidden_size + v_seq * hidden_size + i];
                float weight_v = V_weight[(head_idx * head_dim + tid) * hidden_size + i];
                v_val_pos += input_v * weight_v;
            }
            
            attn_result += weight * v_val_pos;
        }
        
        shared_attn_out[tid] = attn_result;
    }
    __syncthreads();
    
    // Phase 5: Combine all heads and add residual (simplified - assumes single head for clarity)
    if (tid < hidden_size) {
        float residual_val = input[input_offset + tid];
        float attn_val = (tid < head_dim) ? shared_attn_out[tid] : 0.0f;
        float combined = attn_val + residual_val;
        
        // Store in shared memory for layer norm
        if (tid < head_dim) {
            shared_attn_out[tid] = combined;
        }
    }
    __syncthreads();
    
    // Phase 6: Layer Normalization
    if (tid < hidden_size) {
        // Compute mean
        float sum = 0.0f;
        if (tid < head_dim) {
            sum = shared_attn_out[tid];
        }
        
        float mean = block_reduce_sum(sum, reduction_mem) / hidden_size;
        
        // Broadcast mean to all threads
        if (threadIdx.x == 0) {
            reduction_mem[0] = mean;
        }
        __syncthreads();
        mean = reduction_mem[0];
        
        // Compute variance
        float diff = 0.0f;
        if (tid < head_dim) {
            diff = shared_attn_out[tid] - mean;
            diff = diff * diff;
        }
        
        float variance = block_reduce_sum(diff, reduction_mem) / hidden_size;
        
        // Broadcast variance
        if (threadIdx.x == 0) {
            reduction_mem[0] = variance;
        }
        __syncthreads();
        variance = reduction_mem[0];
        
        // Apply layer normalization
        if (tid < head_dim) {
            float normalized = (shared_attn_out[tid] - mean) / sqrtf(variance + eps);
            float ln_w = ln_weight[head_idx * head_dim + tid];
            float ln_b = ln_bias[head_idx * head_dim + tid];
            
            output[input_offset + head_idx * head_dim + tid] = normalized * ln_w + ln_b;
        }
    }
}

// Optimized version with better memory access patterns
__global__ void optimized_fused_attention_layernorm_kernel(
    const float* input,
    const float* qkv_weight,      // Combined Q,K,V weights [3, hidden_size, hidden_size]
    const float* ln_weight,
    const float* ln_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    float eps = 1e-5f
) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / sqrtf(head_dim);
    
    // Shared memory layout
    float* shared_input = shared_mem;
    float* shared_qkv = shared_mem + hidden_size;
    float* shared_attn_out = shared_mem + hidden_size + 3 * hidden_size;
    float* reduction_mem = shared_mem + hidden_size + 3 * hidden_size + hidden_size;
    
    int input_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Load input to shared memory
    if (tid < hidden_size) {
        shared_input[tid] = input[input_offset + tid];
    }
    __syncthreads();
    
    // Compute Q, K, V projections
    for (int proj = 0; proj < 3; proj++) { // Q, K, V
        if (tid < hidden_size) {
            float result = 0.0f;
            for (int i = 0; i < hidden_size; i++) {
                float weight = qkv_weight[proj * hidden_size * hidden_size + tid * hidden_size + i];
                result += shared_input[i] * weight;
            }
            shared_qkv[proj * hidden_size + tid] = result;
        }
        __syncthreads();
    }
    
    // Multi-head attention computation
    float* shared_q = shared_qkv;
    float* shared_k = shared_qkv + hidden_size;
    float* shared_v = shared_qkv + 2 * hidden_size;
    
    // Initialize attention output
    if (tid < hidden_size) {
        shared_attn_out[tid] = 0.0f;
    }
    __syncthreads();
    
    // Process each head
    for (int head = 0; head < num_heads; head++) {
        int head_offset = head * head_dim;
        
        // Compute attention for this head
        if (tid < head_dim) {
            float attn_sum = 0.0f;
            
            // Compute scores and apply softmax
            float scores[MAX_THREADS_PER_BLOCK]; // Assuming seq_len <= MAX_THREADS_PER_BLOCK
            float max_score = -INFINITY;
            
            for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                float score = 0.0f;
                
                // Load K for position k_pos (simplified - should load from global memory)
                for (int d = 0; d < head_dim; d++) {
                    score += shared_q[head_offset + tid] * shared_k[head_offset + d];
                }
                
                scores[k_pos] = score * scale;
                max_score = fmaxf(max_score, scores[k_pos]);
            }
            
            // Apply softmax
            float sum_exp = 0.0f;
            for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                scores[k_pos] = expf(scores[k_pos] - max_score);
                sum_exp += scores[k_pos];
            }
            
            for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                scores[k_pos] = safe_divide(scores[k_pos], sum_exp);
            }
            
            // Compute attention output
            for (int v_pos = 0; v_pos < seq_len; v_pos++) {
                attn_sum += scores[v_pos] * shared_v[head_offset + tid];
            }
            
            shared_attn_out[head_offset + tid] = attn_sum;
        }
    }
    __syncthreads();
    
    // Add residual connection
    if (tid < hidden_size) {
        shared_attn_out[tid] += shared_input[tid];
    }
    __syncthreads();
    
    // Layer normalization
    if (tid < hidden_size) {
        // Compute mean
        float sum = shared_attn_out[tid];
        float mean = block_reduce_sum(sum, reduction_mem) / hidden_size;
        
        // Broadcast mean
        if (threadIdx.x == 0) {
            reduction_mem[0] = mean;
        }
        __syncthreads();
        mean = reduction_mem[0];
        
        // Compute variance
        float diff = shared_attn_out[tid] - mean;
        float variance = block_reduce_sum(diff * diff, reduction_mem) / hidden_size;
        
        // Broadcast variance
        if (threadIdx.x == 0) {
            reduction_mem[0] = variance;
        }
        __syncthreads();
        variance = reduction_mem[0];
        
        // Apply normalization
        float normalized = (shared_attn_out[tid] - mean) / sqrtf(variance + eps);
        output[input_offset + tid] = normalized * ln_weight[tid] + ln_bias[tid];
    }
}

// Host function declarations
extern "C" {
    void launch_fused_attention_layernorm(
        const float* input,
        const float* Q_weight,
        const float* K_weight,
        const float* V_weight,
        const float* ln_weight,
        const float* ln_bias,
        float* output,
        float* temp_scores,
        int batch_size,
        int seq_len,
        int hidden_size,
        int num_heads,
        float eps,
        cudaStream_t stream
    );
    
    void launch_optimized_fused_attention_layernorm(
        const float* input,
        const float* qkv_weight,
        const float* ln_weight,
        const float* ln_bias,
        float* output,
        int batch_size,
        int seq_len,
        int hidden_size,
        int num_heads,
        float eps,
        cudaStream_t stream
    );
}

// Host function implementations
void launch_fused_attention_layernorm(
    const float* input,
    const float* Q_weight,
    const float* K_weight,
    const float* V_weight,
    const float* ln_weight,
    const float* ln_bias,
    float* output,
    float* temp_scores,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    float eps,
    cudaStream_t stream
) {
    int head_dim = hidden_size / num_heads;
    
    // Calculate shared memory requirements
    int shared_mem_size = (3 * head_dim + seq_len + head_dim + 32) * sizeof(float);
    
    dim3 block_size(min(hidden_size, 256));
    dim3 grid_size(num_heads, seq_len, batch_size);
    
    fused_attention_layernorm_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, Q_weight, K_weight, V_weight, ln_weight, ln_bias, output, temp_scores,
        batch_size, seq_len, hidden_size, num_heads, eps
    );
}

void launch_optimized_fused_attention_layernorm(
    const float* input,
    const float* qkv_weight,
    const float* ln_weight,
    const float* ln_bias,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    float eps,
    cudaStream_t stream
) {
    // Calculate shared memory requirements
    int shared_mem_size = (hidden_size + 3 * hidden_size + hidden_size + 64) * sizeof(float);
    
    dim3 block_size(min(hidden_size, 512));
    dim3 grid_size(batch_size, seq_len);
    
    optimized_fused_attention_layernorm_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, qkv_weight, ln_weight, ln_bias, output,
        batch_size, seq_len, hidden_size, num_heads, eps
    );
}
