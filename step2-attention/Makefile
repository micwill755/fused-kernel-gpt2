# Simple Makefile for CUDA Attention Kernel

# CUDA paths (using CUDA 12.8 for compatibility)
CUDA_PATH = /usr/local/cuda-12.8
NVCC = $(CUDA_PATH)/bin/nvcc
INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcublas

# Basic build
attention_test: attention_kernel.cu main.cpp attention_host.cpp
	$(NVCC) -O3 --use_fast_math -arch=sm_86 -std=c++14 $(INCLUDES) $(LIBS) $^ -o $@

# Clean build files
clean:
	rm -f attention_test

# Run the test
run: attention_test
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH ./attention_test

# Multi-GPU architecture support (optional)
multi-arch: attention_kernel.cu main.cpp attention_host.cpp
	$(NVCC) -O3 --use_fast_math -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86 -std=c++14 $(INCLUDES) $(LIBS) $^ -o attention_test

.PHONY: clean run multi-arch
