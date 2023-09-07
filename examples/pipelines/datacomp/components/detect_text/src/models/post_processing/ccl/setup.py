from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ccl_cuda",
    ext_modules=[
        CUDAExtension(
            "ccl_cuda",
            [
                "ccl.cpp",
                "ccl_cuda.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
