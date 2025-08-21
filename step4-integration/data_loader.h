#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <random>

// Simple tokenizer for demo purposes
class SimpleTokenizer {
private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    int vocab_size_;
    
public:
    SimpleTokenizer(int vocab_size = 50257) : vocab_size_(vocab_size) {
        // Create a simple vocabulary for demo
        // In practice, you'd load this from a file (e.g., GPT-2 BPE vocab)
        vocab_.reserve(vocab_size);
        
        // Add special tokens
        vocab_.push_back("<pad>");
        vocab_.push_back("<unk>");
        vocab_.push_back("<bos>");
        vocab_.push_back("<eos>");
        
        // Add common words and characters
        std::vector<std::string> common_tokens = {
            "the", "and", "to", "of", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
            "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
            "but", "not", "what", "all", "were", "we", "when", "your", "can", "said"
        };
        
        for (const auto& token : common_tokens) {
            vocab_.push_back(token);
        }
        
        // Fill remaining vocabulary with dummy tokens
        for (int i = vocab_.size(); i < vocab_size_; i++) {
            vocab_.push_back("token_" + std::to_string(i));
        }
        
        // Build reverse mapping
        for (int i = 0; i < vocab_.size(); i++) {
            token_to_id_[vocab_[i]] = i;
        }
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        
        // Simple word-level tokenization (split by spaces)
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Remove punctuation for simplicity
            word.erase(std::remove_if(word.begin(), word.end(), 
                      [](char c) { return std::ispunct(c); }), word.end());
            
            // Convert to lowercase
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            // Look up token ID
            auto it = token_to_id_.find(word);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(1); // <unk> token
            }
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        
        for (int token_id : tokens) {
            if (token_id >= 0 && token_id < vocab_.size()) {
                if (!text.empty()) text += " ";
                text += vocab_[token_id];
            }
        }
        
        return text;
    }
    
    int getVocabSize() const { return vocab_size_; }
    const std::string& getToken(int id) const { 
        return (id >= 0 && id < vocab_.size()) ? vocab_[id] : vocab_[1]; // Return <unk> for invalid IDs
    }
};

// Dataset class for loading and batching text data
class TextDataset {
private:
    std::vector<std::vector<int>> sequences_;
    std::unique_ptr<SimpleTokenizer> tokenizer_;
    int max_length_;
    
public:
    TextDataset(int max_length = 512) : max_length_(max_length) {
        tokenizer_ = std::make_unique<SimpleTokenizer>();
    }
    
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                auto tokens = tokenizer_->encode(line);
                
                // Truncate or pad to max_length
                if (tokens.size() > max_length_) {
                    tokens.resize(max_length_);
                } else {
                    tokens.resize(max_length_, 0); // Pad with <pad> token
                }
                
                sequences_.push_back(tokens);
            }
        }
        
        std::cout << "Loaded " << sequences_.size() << " sequences from " << filename << std::endl;
    }
    
    void generateSyntheticData(int num_sequences) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, tokenizer_->getVocabSize() - 1);
        
        sequences_.clear();
        sequences_.reserve(num_sequences);
        
        for (int i = 0; i < num_sequences; i++) {
            std::vector<int> sequence(max_length_);
            for (int j = 0; j < max_length_; j++) {
                sequence[j] = dis(gen);
            }
            sequences_.push_back(sequence);
        }
        
        std::cout << "Generated " << num_sequences << " synthetic sequences" << std::endl;
    }
    
    struct Batch {
        std::vector<int> input_ids;
        int batch_size;
        int seq_length;
        
        Batch(int bs, int sl) : batch_size(bs), seq_length(sl) {
            input_ids.resize(batch_size * seq_length);
        }
    };
    
    Batch getBatch(int batch_size, int start_idx = -1) {
        if (sequences_.empty()) {
            throw std::runtime_error("No data loaded");
        }
        
        Batch batch(batch_size, max_length_);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, sequences_.size() - 1);
        
        for (int i = 0; i < batch_size; i++) {
            int seq_idx = (start_idx >= 0) ? (start_idx + i) % sequences_.size() : dis(gen);
            
            for (int j = 0; j < max_length_; j++) {
                batch.input_ids[i * max_length_ + j] = sequences_[seq_idx][j];
            }
        }
        
        return batch;
    }
    
    int size() const { return sequences_.size(); }
    int getMaxLength() const { return max_length_; }
    SimpleTokenizer& getTokenizer() { return *tokenizer_; }
};

// Data loader for training/evaluation
class DataLoader {
private:
    std::unique_ptr<TextDataset> dataset_;
    int batch_size_;
    bool shuffle_;
    std::vector<int> indices_;
    int current_idx_;
    
public:
    DataLoader(std::unique_ptr<TextDataset> dataset, int batch_size, bool shuffle = true)
        : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle), current_idx_(0) {
        
        // Initialize indices
        indices_.resize(dataset_->size());
        std::iota(indices_.begin(), indices_.end(), 0);
        
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
    }
    
    bool hasNext() const {
        return current_idx_ < indices_.size();
    }
    
    TextDataset::Batch getNextBatch() {
        if (!hasNext()) {
            throw std::runtime_error("No more batches available");
        }
        
        int actual_batch_size = std::min(batch_size_, (int)indices_.size() - current_idx_);
        auto batch = dataset_->getBatch(actual_batch_size, indices_[current_idx_]);
        
        current_idx_ += actual_batch_size;
        
        return batch;
    }
    
    void reset() {
        current_idx_ = 0;
        
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
    }
    
    int getBatchSize() const { return batch_size_; }
    int getNumBatches() const { return (dataset_->size() + batch_size_ - 1) / batch_size_; }
    TextDataset& getDataset() { return *dataset_; }
};

// Utility functions for data processing
namespace DataUtils {
    
    // Create a simple text file for testing
    void createSampleTextFile(const std::string& filename, int num_lines = 1000) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not create file: " + filename);
        }
        
        std::vector<std::string> sample_sentences = {
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is transforming the world",
            "Machine learning models require large amounts of data",
            "CUDA programming enables high performance computing",
            "Deep learning has revolutionized computer vision",
            "Natural language processing helps computers understand text",
            "GPT models can generate human-like text",
            "Transformer architectures use attention mechanisms",
            "Parallel computing accelerates neural network training",
            "GPU kernels optimize memory bandwidth utilization"
        };
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, sample_sentences.size() - 1);
        
        for (int i = 0; i < num_lines; i++) {
            file << sample_sentences[dis(gen)] << std::endl;
        }
        
        std::cout << "Created sample text file: " << filename << " with " << num_lines << " lines" << std::endl;
    }
    
    // Load pre-trained tokenizer vocabulary (placeholder)
    std::unique_ptr<SimpleTokenizer> loadPretrainedTokenizer(const std::string& vocab_file) {
        // In a real implementation, this would load GPT-2 BPE vocabulary
        // For now, return a simple tokenizer
        return std::make_unique<SimpleTokenizer>();
    }
    
    // Calculate dataset statistics
    struct DatasetStats {
        int total_sequences;
        int total_tokens;
        double avg_sequence_length;
        int vocab_size;
        std::vector<int> token_frequencies;
    };
    
    DatasetStats calculateStats(const TextDataset& dataset) {
        DatasetStats stats;
        stats.total_sequences = dataset.size();
        stats.vocab_size = dataset.getTokenizer().getVocabSize();
        stats.token_frequencies.resize(stats.vocab_size, 0);
        
        // This is a simplified version - in practice you'd iterate through actual data
        stats.total_tokens = stats.total_sequences * dataset.getMaxLength();
        stats.avg_sequence_length = dataset.getMaxLength();
        
        return stats;
    }
    
    void printDatasetStats(const DatasetStats& stats) {
        std::cout << "Dataset Statistics:" << std::endl;
        std::cout << "  Total sequences: " << stats.total_sequences << std::endl;
        std::cout << "  Total tokens: " << stats.total_tokens << std::endl;
        std::cout << "  Average sequence length: " << stats.avg_sequence_length << std::endl;
        std::cout << "  Vocabulary size: " << stats.vocab_size << std::endl;
    }
}

#endif // DATA_LOADER_H
