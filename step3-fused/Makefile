# Makefile for CUDA Fused Kernel

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86

# Compiler flags
NVCC_FLAGS = -O3 --use_fast_math -Xptxas=-v $(CUDA_ARCH) --expt-relaxed-constexpr
CXX_FLAGS = -O3 -std=c++14

# Include directories
INCLUDES = -I/usr/local/cuda/include -I.

# Library directories and libraries
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

# Source files
CUDA_SOURCES = fused_kernel.cu
CPP_SOURCES = main.cpp
HEADERS = benchmark.h
OBJECTS = fused_kernel.o main.o

# Target executable
TARGET = fused_test

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile CUDA source files
fused_kernel.o: fused_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c fused_kernel.cu -o fused_kernel.o

# Compile C++ source files
main.o: main.cpp $(HEADERS)
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

# Profile build with line info
profile: NVCC_FLAGS += -lineinfo
profile: $(TARGET)

# Optimized build with maximum performance
optimize: NVCC_FLAGS += -maxrregcount=64 --ptxas-options=-v
optimize: $(TARGET)

# Check CUDA installation and GPU info
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@echo ""
	@echo "GPU Information:"
	@nvidia-smi

# Memory check with cuda-memcheck
memcheck: $(TARGET)
	cuda-memcheck ./$(TARGET)

# Profile with nsight-compute (if available)
nsight-profile: $(TARGET)
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --metrics dram__bytes_read,dram__bytes_write,sm__cycles_elapsed ./$(TARGET); \
	else \
		echo "NVIDIA Nsight Compute not found. Install it for detailed profiling."; \
	fi

# Benchmark with different configurations
benchmark: $(TARGET)
	@echo "Running comprehensive benchmarks..."
	@echo "=================================="
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the fused kernel test program (default)"
	@echo "  clean       - Remove build files"
	@echo "  run         - Build and run the test program"
	@echo "  debug       - Build with debug symbols"
	@echo "  profile     - Build with profiling information"
	@echo "  optimize    - Build with maximum optimization"
	@echo "  check-cuda  - Check CUDA installation and GPU info"
	@echo "  memcheck    - Run with CUDA memory checker"
	@echo "  nsight-profile - Profile with NVIDIA Nsight Compute"
	@echo "  benchmark   - Run comprehensive benchmarks"
	@echo "  help        - Show this help message"

.PHONY: all clean run debug profile optimize check-cuda memcheck nsight-profile benchmark help
