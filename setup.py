# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from Cython.Build import cythonize

import numpy as np

from slopescreening import __version__

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='slope-screening',
    version=__version__,
    description='Safe screening rules for the SLOPE problem',
    long_description=readme,
    author='Clément Elvira and Cédric Herzet',
    author_email='clement.elvira@centralesupelec.fr',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    ext_modules=cythonize([
        "slopescreening/screening/utils.pyx", 
        "slopescreening/screening/gap_test_all.pyx",
        "slopescreening/screening/gap_test_p_1.pyx",
    ]),
    include_dirs=[np.get_include()],
)
