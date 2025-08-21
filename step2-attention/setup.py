from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install CUDA and PyTorch with CUDA support.")

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building with CUDA {cuda_version}")

# CUDA compilation flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-Xptxas=-v',
    '--expt-relaxed-constexpr',
    '-gencode=arch=compute_70,code=sm_70',  # V100
    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
    '-gencode=arch=compute_80,code=sm_80',  # A100
    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
    '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
]

# C++ compilation flags
cxx_flags = [
    '-O3',
    '-std=c++14',
]

setup(
    name='attention_kernel',
    ext_modules=[
        CUDAExtension(
            name='attention_kernel',
            sources=[
                'attention_host.cpp',
                'attention_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags
            },
            include_dirs=[
                # Add any additional include directories here
            ],
            libraries=[
                'cudart',
                'cublas',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'numpy',
    ],
)
