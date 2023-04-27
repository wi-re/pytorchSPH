from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='neighborSearch',
    ext_modules=[
        CppExtension('neighborSearch', ['neighSearch.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 