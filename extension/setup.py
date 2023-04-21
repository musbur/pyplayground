import os
from setuptools import setup, Extension

e1 = Extension('tuple_list', sources=['tuple_list.c'])

setup(
    name='some_tests',
    ext_modules=[e1],
)

def link_extensions(lpath):
    """
    Walk recursively through the 'build' directory and symlink any '.so'
    file into the source tree so that the package is locally functional
    after "python setup.py build" instead of having to do "pip install -e ."
    """
    dir = os.path.join(*lpath)
    for f in os.scandir(dir):
        if f.is_dir():
            link_extensions(lpath + [f.name])
        elif f.is_file and f.name.endswith('.so'):
            dirpath = os.path.join(*lpath[2:], '.')
            relpath = os.path.join('.', *(['..'] * (len(lpath) - 2)),
                                   *lpath,
                                   f.name)
            dir_fd = os.open(dirpath, os.O_RDONLY)
            try:
                os.symlink(relpath, f.name, dir_fd=dir_fd)
            except FileExistsError:
                pass

if os.path.exists('build'):
    link_extensions(['build'])
