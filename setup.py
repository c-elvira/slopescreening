# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import numpy as np

setup(
    packages=find_packages(exclude=('tests', 'docs')),
    cmdclass={'build_ext': build_ext},
    # ext_modules=cythonize([
    #     "slopescreening/screening/utils.pyx",
    #     "slopescreening/screening/gap_test_all.pyx",
    #     "slopescreening/screening/gap_test_p_1.pyx",
    # ]),
    ext_modules=[
        Extension('slopescreening.screening.utils',
                  sources=['slopescreening/screening/utils.pyx'],
                  language='c++',
                  include_dirs=[np.get_include()],
                  extra_compile_args=["-O3"]),
        Extension('slopescreening.screening.gap_test_all',
                  sources=['slopescreening/screening/gap_test_all.pyx'],
                  language='c++',
                  include_dirs=[np.get_include()],
                  extra_compile_args=["-O3"]),
        Extension('slopescreening.screening.gap_test_p_1',
                  sources=['slopescreening/screening/gap_test_p_1.pyx'],
                  language='c++',
                  include_dirs=[np.get_include()],
                  extra_compile_args=["-O3"]),
    ],
    include_dirs=[np.get_include()],
)
