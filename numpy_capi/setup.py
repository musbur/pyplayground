from setuptools import setup, Extension

from numpy.distutils.misc_util import get_numpy_include_dirs
from numpy import __version__ as numpy_version

numpy_cflags = ["-DWITH_NUMPY=" + numpy_version] \
    + ["-I"+dd for dd in get_numpy_include_dirs()]

ex_pto = Extension('pto.pto_csv',
                             sources=['pto/pypto_csv.c',
                                      'pto/pto_csv.c',
                                      'pto/csv_read.c'],
                             extra_compile_args=numpy_cflags)
setup (name = 'test_numpy_capi',
       version = '0.1',
       ext_modules = [ex_pto],
       packages = [])

