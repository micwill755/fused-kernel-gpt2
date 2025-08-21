# Makefile for CUDA Attention Kernel

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86

# Compiler flags
NVCC_FLAGS = -O3 --use_fast_math -Xptxas=-v $(CUDA_ARCH)
CXX_FLAGS = -O3 -std=c++14

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories and libraries
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

# Source files
CUDA_SOURCES = attention_kernel.cu
CPP_SOURCES = main.cpp
OBJECTS = attention_kernel.o main.o

# Target executable
TARGET = attention_test

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile CUDA source files
attention_kernel.o: attention_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c attention_kernel.cu -o attention_kernel.o

# Compile C++ source files
main.o: main.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c main.cpp -o main.o

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the test
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -lineinfo
profile: $(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@nvidia-smi

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the attention test program (default)"
	@echo "  clean    - Remove build files"
	@echo "  run      - Build and run the test program"
	@echo "  debug    - Build with debug symbols"
	@echo "  profile  - Build with profiling information"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  help     - Show this help message"

.PHONY: all clean run debug profile check-cuda help
