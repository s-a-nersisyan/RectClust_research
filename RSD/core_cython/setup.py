from setuptools import setup
from Cython.Build import cythonize

import numpy as np

setup(ext_modules=cythonize("MLE.pyx", annotate=True), include_dirs=[np.get_include()])
setup(ext_modules=cythonize("EM_utils.pyx", annotate=True), include_dirs=[np.get_include()])
