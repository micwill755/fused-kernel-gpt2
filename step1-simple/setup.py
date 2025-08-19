from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='simple_kernel',
    ext_modules=[
        CUDAExtension(
            name='simple_kernel',
            sources=[
                'simple_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',  # RTX 20xx series
                    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx series  
                    '-gencode=arch=compute_80,code=sm_80',  # RTX 30xx series
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx series
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
