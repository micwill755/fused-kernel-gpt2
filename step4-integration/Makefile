# Makefile for GPT-2 Integration

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPU)
CUDA_ARCH = -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86

# Compiler flags
NVCC_FLAGS = -O3 --use_fast_math -Xptxas=-v $(CUDA_ARCH) --expt-relaxed-constexpr
CXX_FLAGS = -O3 -std=c++14 -Wall -Wextra

# Include directories
INCLUDES = -I/usr/local/cuda/include -I. -I../step3-fused

# Library directories and libraries
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand

# Source files
CUDA_SOURCES = gpt2_model.cu ../step3-fused/fused_kernel.cu
CPP_SOURCES = main.cpp
HEADERS = gpt2_model.h data_loader.h ../step3-fused/benchmark.h
OBJECTS = gpt2_model.o fused_kernel.o main.o

# Target executable
TARGET = gpt2_demo

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile CUDA source files
gpt2_model.o: gpt2_model.cu gpt2_model.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c gpt2_model.cu -o gpt2_model.o

# Compile fused kernel from step3
fused_kernel.o: ../step3-fused/fused_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c ../step3-fused/fused_kernel.cu -o fused_kernel.o

# Compile C++ source files
main.o: main.cpp $(HEADERS)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c main.cpp -o main.o

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the demo
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
optimize: CXX_FLAGS += -march=native -mtune=native
optimize: $(TARGET)

# Build with different model sizes
tiny: NVCC_FLAGS += -DMODEL_SIZE=TINY
tiny: $(TARGET)

small: NVCC_FLAGS += -DMODEL_SIZE=SMALL
small: $(TARGET)

medium: NVCC_FLAGS += -DMODEL_SIZE=MEDIUM
medium: $(TARGET)

# Check CUDA installation and GPU info
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@echo ""
	@echo "GPU Information:"
	@nvidia-smi

# Memory check with cuda-memcheck
memcheck: $(TARGET)
	cuda-memcheck --tool memcheck ./$(TARGET)

# Race condition check
racecheck: $(TARGET)
	cuda-memcheck --tool racecheck ./$(TARGET)

# Profile with nsight-compute (if available)
nsight-profile: $(TARGET)
	@if command -v ncu >/dev/null 2>&1; then \
		ncu --metrics dram__bytes_read,dram__bytes_write,sm__cycles_elapsed,sm__warps_active ./$(TARGET); \
	else \
		echo "NVIDIA Nsight Compute not found. Install it for detailed profiling."; \
	fi

# Profile with nsight-systems (if available)
nsight-systems: $(TARGET)
	@if command -v nsys >/dev/null 2>&1; then \
		nsys profile --stats=true ./$(TARGET); \
	else \
		echo "NVIDIA Nsight Systems not found. Install it for timeline profiling."; \
	fi

# Benchmark with different configurations
benchmark: $(TARGET)
	@echo "Running comprehensive benchmarks..."
	@echo "=================================="
	./$(TARGET)

# Test with synthetic data
test-synthetic: $(TARGET)
	@echo "Testing with synthetic data..."
	./$(TARGET) --synthetic

# Create sample data file
sample-data:
	@echo "Creating sample text data..."
	@echo "The quick brown fox jumps over the lazy dog" > sample.txt
	@echo "Artificial intelligence is transforming the world" >> sample.txt
	@echo "Machine learning models require large amounts of data" >> sample.txt
	@echo "CUDA programming enables high performance computing" >> sample.txt
	@echo "Deep learning has revolutionized computer vision" >> sample.txt
	@echo "Sample data created in sample.txt"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake
	@echo "Make sure CUDA Toolkit is installed separately"

# Check for memory leaks with valgrind (CPU parts only)
valgrind: $(TARGET)
	@if command -v valgrind >/dev/null 2>&1; then \
		valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET); \
	else \
		echo "Valgrind not found. Install it for memory leak detection."; \
	fi

# Generate documentation (if doxygen is available)
docs:
	@if command -v doxygen >/dev/null 2>&1; then \
		doxygen Doxyfile; \
	else \
		echo "Doxygen not found. Install it for documentation generation."; \
	fi

# Performance comparison with different optimization levels
perf-compare:
	@echo "Building with different optimization levels..."
	$(MAKE) clean
	$(MAKE) NVCC_FLAGS="-O0 $(CUDA_ARCH)" CXX_FLAGS="-O0 -std=c++14" TARGET=gpt2_demo_O0
	$(MAKE) clean
	$(MAKE) NVCC_FLAGS="-O2 $(CUDA_ARCH)" CXX_FLAGS="-O2 -std=c++14" TARGET=gpt2_demo_O2
	$(MAKE) clean
	$(MAKE) optimize TARGET=gpt2_demo_O3
	@echo "Performance comparison executables created: gpt2_demo_O0, gpt2_demo_O2, gpt2_demo_O3"

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build the GPT-2 demo program (default)"
	@echo "  clean         - Remove build files"
	@echo "  run           - Build and run the demo program"
	@echo "  debug         - Build with debug symbols"
	@echo "  profile       - Build with profiling information"
	@echo "  optimize      - Build with maximum optimization"
	@echo "  tiny/small/medium - Build with specific model sizes"
	@echo "  check-cuda    - Check CUDA installation and GPU info"
	@echo "  memcheck      - Run with CUDA memory checker"
	@echo "  racecheck     - Run with CUDA race condition checker"
	@echo "  nsight-profile - Profile with NVIDIA Nsight Compute"
	@echo "  nsight-systems - Profile with NVIDIA Nsight Systems"
	@echo "  benchmark     - Run comprehensive benchmarks"
	@echo "  test-synthetic - Test with synthetic data"
	@echo "  sample-data   - Create sample text data file"
	@echo "  install-deps  - Install system dependencies"
	@echo "  valgrind      - Check for memory leaks (CPU parts)"
	@echo "  docs          - Generate documentation"
	@echo "  perf-compare  - Build with different optimization levels"
	@echo "  help          - Show this help message"

.PHONY: all clean run debug profile optimize tiny small medium check-cuda memcheck racecheck nsight-profile nsight-systems benchmark test-synthetic sample-data install-deps valgrind docs perf-compare help
