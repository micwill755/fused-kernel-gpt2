# Environment Setup

## System Requirements

- **OS**: Linux or macOS (Windows with WSL2)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (RTX 20xx series or newer)
- **CUDA**: Version 11.8 or 12.x
- **Python**: 3.8-3.11

## Installation Steps

### 1. Check CUDA Installation
```bash
nvcc --version
nvidia-smi
```

### 2. Create Conda Environment
```bash
conda create -n kernel-fusion python=3.9
conda activate kernel-fusion
```

### 3. Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Development Tools
```bash
pip install ninja
pip install pybind11
conda install -c conda-forge cudatoolkit-dev
```

### 5. Verify Installation
```python3
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Common Issues

**Issue**: `nvcc not found`
**Solution**: Add CUDA to PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue**: `torch.cuda.is_available()` returns False
**Solution**: Reinstall PyTorch with correct CUDA version

**Issue**: Compilation errors
**Solution**: Check GCC compatibility with CUDA version

### Recommended Development Setup

**IDE**: VSCode with extensions:
- C/C++
- Python
- CUDA C++
- Better Comments

**Debugging Tools**:
- `cuda-gdb` for kernel debugging
- `nsight-compute` for profiling
- `nvprof` for basic profiling

## Performance Monitoring

Install profiling tools:
```bash
# For detailed kernel analysis
sudo apt install nvidia-nsight-compute

# For system monitoring
pip install gpustat
pip install nvidia-ml-py3
```

## Next Step
Once your environment is set up, proceed to `step1-simple/` to start with your first CUDA kernel.
