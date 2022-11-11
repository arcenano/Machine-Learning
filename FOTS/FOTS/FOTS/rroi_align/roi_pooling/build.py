
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup, Extension, find_packages


def get_extensions():
    sources = ['roi_pooling.c']

    ext_modules = [
            CppExtension(
                "rroi_align",
                sources,
                headers='roi_pooling.h',
                with_cuda=False,
                include_dirs=["/owner/.conda/envs/torch/lib/python3.9/site-packages/torch/include"],
                #define_macros=define_macros,
                #extra_compile_args=extra_compile_args,
            )
        ]
    return ext_modules


if __name__ == '__main__':
    setup(
    name="rroi_align",
    # version="0.1",
    # author="fmassa",
    # url="https://github.com/facebookresearch/maskrcnn-benchmark",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension})