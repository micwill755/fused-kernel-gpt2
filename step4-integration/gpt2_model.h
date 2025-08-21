#ifndef GPT2_MODEL_H
#define GPT2_MODEL_H

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

// Forward declarations
class Matrix;
class GPT2Config;
class GPT2Block;

// Configuration structure for GPT-2 model
struct GPT2Config {
    int vocab_size = 50257;
    int max_position_embeddings = 1024;
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072;
    float layer_norm_epsilon = 1e-5f;
    float attention_dropout = 0.1f;
    float hidden_dropout = 0.1f;
    
    GPT2Config() = default;
    GPT2Config(int vocab, int max_pos, int hidden, int layers, int heads, int intermediate)
        : vocab_size(vocab), max_position_embeddings(max_pos), hidden_size(hidden),
          num_hidden_layers(layers), num_attention_heads(heads), intermediate_size(intermediate) {}
};

// Matrix class for GPU operations
class Matrix {
private:
    float* data_;
    int rows_;
    int cols_;
    bool owns_data_;
    
public:
    Matrix(int rows, int cols, bool allocate_gpu = true);
    Matrix(float* data, int rows, int cols); // Wrap existing data
    ~Matrix();
    
    // Copy constructor and assignment operator
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    
    // Move constructor and assignment operator
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    
    // Accessors
    float* data() { return data_; }
    const float* data() const { return data_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    
    // Operations
    void zero();
    void fill(float value);
    void randomize(float mean = 0.0f, float std = 0.02f);
    void copyFromHost(const float* host_data);
    void copyToHost(float* host_data) const;
    
    // Matrix operations
    void matmul(const Matrix& a, const Matrix& b, bool transpose_b = false);
    void add(const Matrix& other);
    void scale(float factor);
    
    // Utility functions
    void print(const std::string& name = "", int max_elements = 10) const;
};

// Embedding layer
class Embedding {
private:
    Matrix weight_;
    int vocab_size_;
    int embedding_dim_;
    
public:
    Embedding(int vocab_size, int embedding_dim);
    
    void forward(const int* input_ids, int batch_size, int seq_len, Matrix& output);
    void randomizeWeights();
    
    Matrix& getWeight() { return weight_; }
};

// Layer normalization
class LayerNorm {
private:
    Matrix weight_;
    Matrix bias_;
    int hidden_size_;
    float eps_;
    
public:
    LayerNorm(int hidden_size, float eps = 1e-5f);
    
    void forward(const Matrix& input, Matrix& output);
    void randomizeWeights();
    
    Matrix& getWeight() { return weight_; }
    Matrix& getBias() { return bias_; }
};

// Multi-layer perceptron (feed-forward network)
class MLP {
private:
    Matrix fc1_weight_;
    Matrix fc1_bias_;
    Matrix fc2_weight_;
    Matrix fc2_bias_;
    int hidden_size_;
    int intermediate_size_;
    
public:
    MLP(int hidden_size, int intermediate_size);
    
    void forward(const Matrix& input, Matrix& output, Matrix& temp);
    void randomizeWeights();
};

// Fused attention + layer norm block
class FusedAttentionLayerNorm {
private:
    Matrix q_weight_;
    Matrix k_weight_;
    Matrix v_weight_;
    Matrix output_weight_;
    Matrix output_bias_;
    Matrix ln_weight_;
    Matrix ln_bias_;
    int hidden_size_;
    int num_heads_;
    float eps_;
    
public:
    FusedAttentionLayerNorm(int hidden_size, int num_heads, float eps = 1e-5f);
    
    void forward(const Matrix& input, Matrix& output, Matrix& temp_scores);
    void randomizeWeights();
    
    // Accessors for weights (for loading pre-trained models)
    Matrix& getQWeight() { return q_weight_; }
    Matrix& getKWeight() { return k_weight_; }
    Matrix& getVWeight() { return v_weight_; }
    Matrix& getOutputWeight() { return output_weight_; }
    Matrix& getOutputBias() { return output_bias_; }
    Matrix& getLNWeight() { return ln_weight_; }
    Matrix& getLNBias() { return ln_bias_; }
};

// GPT-2 transformer block
class GPT2Block {
private:
    std::unique_ptr<FusedAttentionLayerNorm> fused_attn_ln_;
    std::unique_ptr<LayerNorm> ln2_;
    std::unique_ptr<MLP> mlp_;
    int hidden_size_;
    int num_heads_;
    int intermediate_size_;
    
public:
    GPT2Block(int hidden_size, int num_heads, int intermediate_size);
    
    void forward(const Matrix& input, Matrix& output, Matrix& temp1, Matrix& temp2, Matrix& temp_scores);
    void randomizeWeights();
    
    // Accessors
    FusedAttentionLayerNorm& getFusedAttentionLayerNorm() { return *fused_attn_ln_; }
    LayerNorm& getLayerNorm2() { return *ln2_; }
    MLP& getMLP() { return *mlp_; }
};

// Main GPT-2 model
class GPT2Model {
private:
    GPT2Config config_;
    std::unique_ptr<Embedding> token_embedding_;
    std::unique_ptr<Embedding> position_embedding_;
    std::vector<std::unique_ptr<GPT2Block>> blocks_;
    std::unique_ptr<LayerNorm> final_layer_norm_;
    
    // Temporary matrices for computation
    std::vector<Matrix> temp_matrices_;
    Matrix temp_scores_;
    
public:
    GPT2Model(const GPT2Config& config);
    
    void forward(const int* input_ids, int batch_size, int seq_len, Matrix& output);
    void randomizeWeights();
    
    // Model management
    void saveWeights(const std::string& filename) const;
    void loadWeights(const std::string& filename);
    
    // Accessors
    const GPT2Config& getConfig() const { return config_; }
    GPT2Block& getBlock(int index) { return *blocks_[index]; }
    Embedding& getTokenEmbedding() { return *token_embedding_; }
    Embedding& getPositionEmbedding() { return *position_embedding_; }
    LayerNorm& getFinalLayerNorm() { return *final_layer_norm_; }
    
    // Utility functions
    void printModelInfo() const;
    size_t getParameterCount() const;
    size_t getMemoryUsage() const;
};

// Language modeling head (for text generation)
class GPT2LMHeadModel {
private:
    std::unique_ptr<GPT2Model> transformer_;
    Matrix lm_head_weight_;
    GPT2Config config_;
    
public:
    GPT2LMHeadModel(const GPT2Config& config);
    
    void forward(const int* input_ids, int batch_size, int seq_len, Matrix& logits);
    void randomizeWeights();
    
    // Text generation
    std::vector<int> generate(const std::vector<int>& input_ids, int max_length, 
                             float temperature = 1.0f, int top_k = 50);
    
    // Model management
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);
    
    // Accessors
    GPT2Model& getTransformer() { return *transformer_; }
    Matrix& getLMHeadWeight() { return lm_head_weight_; }
    const GPT2Config& getConfig() const { return config_; }
};

// Utility functions
namespace GPT2Utils {
    // Initialize CUDA and check for errors
    void initializeCUDA();
    
    // Memory management helpers
    void* allocateGPUMemory(size_t bytes);
    void freeGPUMemory(void* ptr);
    
    // Random number generation
    void setRandomSeed(unsigned int seed);
    
    // Tokenization (simple word-level for demo)
    std::vector<int> tokenize(const std::string& text, int vocab_size = 50257);
    std::string detokenize(const std::vector<int>& tokens);
    
    // Model loading helpers
    void loadPretrainedWeights(GPT2LMHeadModel& model, const std::string& model_name);
    
    // Benchmarking helpers
    float benchmarkInference(GPT2LMHeadModel& model, int batch_size, int seq_len, int iterations = 100);
    void profileMemoryUsage(GPT2LMHeadModel& model, int batch_size, int seq_len);
}

#endif // GPT2_MODEL_H
