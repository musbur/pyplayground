from setuptools import setup, Extension

from numpy.distutils.misc_util import get_numpy_include_dirs
from numpy import __version__ as numpy_version
numpy_cflags = ["-DWITH_NUMPY=" + numpy_version] \
    + ["-I"+dd for dd in get_numpy_include_dirs()]

numpy_capi = Extension('numpy_capi',
                       sources=['numpy_capi.c'],
                       extra_compile_args=numpy_cflags)

numpy_capi_class = Extension('numpy_capi_class',
                             sources=['pto/numpy_capi_class.c',
                                      'pto/read_csv.c'],
                             extra_compile_args=numpy_cflags)
setup (name = 'test_numpy_capi',
       version = '0.1',
       ext_modules = [numpy_capi, numpy_capi_class],
       packages = [])

