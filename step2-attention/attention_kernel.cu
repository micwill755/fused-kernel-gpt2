#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#define TILE_SIZE 16
#define MAX_SEQ_LEN 1024
#define WARP_SIZE 32

// Utility function for safe division
__device__ __forceinline__ float safe_divide(float a, float b) {
    return b == 0.0f ? 0.0f : a / b;
}

// Softmax kernel with numerical stability
__global__ void softmax_kernel(float* input, float* output, int batch_size, int seq_len, int hidden_size) {
    int batch_idx = blockIdx.x;
    int seq_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // Find maximum value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        max_val = fmaxf(max_val, input[idx]);
    }
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        float exp_val = expf(input[idx] - max_val);
        output[idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < seq_len; i++) {
        int idx = batch_idx * seq_len * seq_len + seq_idx * seq_len + i;
        output[idx] = safe_divide(output[idx], sum);
    }
}

// Optimized matrix multiplication with shared memory
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K, bool transpose_B = false) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            As[ty][tx] = A[a_row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B (with optional transpose)
        int b_row, b_col;
        if (transpose_B) {
            b_row = col;
            b_col = tile * TILE_SIZE + ty;
            if (b_row < N && b_col < K) {
                Bs[ty][tx] = B[b_row * K + b_col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        } else {
            b_row = tile * TILE_SIZE + ty;
            b_col = col;
            if (b_row < K && b_col < N) {
                Bs[ty][tx] = B[b_row * N + b_col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Multi-head attention kernel
__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                float* output, float* temp_scores,
                                int batch_size, int seq_len, int hidden_size,
                                int num_heads, float scale_factor) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    int head_dim = hidden_size / num_heads;
    int q_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + head_idx * head_dim;
    int k_offset = batch_idx * seq_len * hidden_size + head_idx * head_dim;
    int v_offset = batch_idx * seq_len * hidden_size + head_idx * head_dim;
    int score_offset = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + seq_idx * seq_len;
    
    // Compute attention scores: Q @ K^T
    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q[q_offset + d];
            float k_val = K[k_offset + k_seq * hidden_size + d];
            score += q_val * k_val;
        }
        temp_scores[score_offset + k_seq] = score * scale_factor;
    }
    
    __syncthreads();
    
    // Apply softmax (simplified version - in practice, use shared memory optimization)
    float max_score = -INFINITY;
    for (int k = 0; k < seq_len; k++) {
        max_score = fmaxf(max_score, temp_scores[score_offset + k]);
    }
    
    float sum_exp = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        float exp_score = expf(temp_scores[score_offset + k] - max_score);
        temp_scores[score_offset + k] = exp_score;
        sum_exp += exp_score;
    }
    
    for (int k = 0; k < seq_len; k++) {
        temp_scores[score_offset + k] = safe_divide(temp_scores[score_offset + k], sum_exp);
    }
    
    __syncthreads();
    
    // Compute final output: attention_weights @ V
    int out_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float result = 0.0f;
        for (int v_seq = 0; v_seq < seq_len; v_seq++) {
            float weight = temp_scores[score_offset + v_seq];
            float v_val = V[v_offset + v_seq * hidden_size + d];
            result += weight * v_val;
        }
        output[out_offset + d] = result;
    }
}

// Optimized attention kernel with shared memory
__global__ void optimized_attention_kernel(const float* Q, const float* K, const float* V,
                                          float* output, float* temp_scores,
                                          int batch_size, int seq_len, int hidden_size,
                                          int num_heads, float scale_factor) {
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int head_dim = hidden_size / num_heads;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    // Shared memory layout
    float* shared_q = shared_mem;
    float* shared_k = shared_mem + head_dim;
    float* shared_v = shared_mem + 2 * head_dim;
    float* shared_scores = shared_mem + 3 * head_dim;
    
    // Process each query position
    for (int q_pos = 0; q_pos < seq_len; q_pos++) {
        // Load Q vector for current position
        if (tid < head_dim) {
            int q_idx = batch_idx * seq_len * hidden_size + q_pos * hidden_size + head_idx * head_dim + tid;
            shared_q[tid] = Q[q_idx];
        }
        __syncthreads();
        
        // Compute scores for all key positions
        for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
            float score = 0.0f;
            
            // Load K vector
            for (int d = 0; d < head_dim; d++) {
                int k_idx = batch_idx * seq_len * hidden_size + k_pos * hidden_size + head_idx * head_dim + d;
                score += shared_q[d] * K[k_idx];
            }
            
            shared_scores[k_pos] = score * scale_factor;
        }
        __syncthreads();
        
        // Softmax reduction
        if (tid == 0) {
            float max_score = -INFINITY;
            for (int i = 0; i < seq_len; i++) {
                max_score = fmaxf(max_score, shared_scores[i]);
            }
            
            float sum_exp = 0.0f;
            for (int i = 0; i < seq_len; i++) {
                shared_scores[i] = expf(shared_scores[i] - max_score);
                sum_exp += shared_scores[i];
            }
            
            for (int i = 0; i < seq_len; i++) {
                shared_scores[i] = safe_divide(shared_scores[i], sum_exp);
            }
        }
        __syncthreads();
        
        // Compute output
        if (tid < head_dim) {
            float result = 0.0f;
            for (int v_pos = 0; v_pos < seq_len; v_pos++) {
                int v_idx = batch_idx * seq_len * hidden_size + v_pos * hidden_size + head_idx * head_dim + tid;
                result += shared_scores[v_pos] * V[v_idx];
            }
            
            int out_idx = batch_idx * seq_len * hidden_size + q_pos * hidden_size + head_idx * head_dim + tid;
            output[out_idx] = result;
        }
        __syncthreads();
    }
}

// Host function declarations
extern "C" {
    void launch_attention_kernel(const float* Q, const float* K, const float* V,
                                float* output, float* temp_scores,
                                int batch_size, int seq_len, int hidden_size,
                                int num_heads, cudaStream_t stream);
    
    void launch_optimized_attention_kernel(const float* Q, const float* K, const float* V,
                                          float* output, float* temp_scores,
                                          int batch_size, int seq_len, int hidden_size,
                                          int num_heads, cudaStream_t stream);
}

// Host function implementations
void launch_attention_kernel(const float* Q, const float* K, const float* V,
                            float* output, float* temp_scores,
                            int batch_size, int seq_len, int hidden_size,
                            int num_heads, cudaStream_t stream) {
    float scale_factor = 1.0f / sqrtf(hidden_size / num_heads);
    
    dim3 block_size(32);
    dim3 grid_size((seq_len + block_size.x - 1) / block_size.x, num_heads, batch_size);
    
    attention_kernel<<<grid_size, block_size, 0, stream>>>(
        Q, K, V, output, temp_scores,
        batch_size, seq_len, hidden_size, num_heads, scale_factor
    );
}

void launch_optimized_attention_kernel(const float* Q, const float* K, const float* V,
                                      float* output, float* temp_scores,
                                      int batch_size, int seq_len, int hidden_size,
                                      int num_heads, cudaStream_t stream) {
    float scale_factor = 1.0f / sqrtf(hidden_size / num_heads);
    int head_dim = hidden_size / num_heads;
    
    // Calculate shared memory size
    int shared_mem_size = (3 * head_dim + seq_len) * sizeof(float);
    
    dim3 block_size(min(head_dim, 256));
    dim3 grid_size(1, num_heads, batch_size);
    
    optimized_attention_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        Q, K, V, output, temp_scores,
        batch_size, seq_len, hidden_size, num_heads, scale_factor
    );
}
