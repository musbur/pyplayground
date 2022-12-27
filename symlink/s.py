import os

dir_fd = os.open('bar', os.O_RDONLY)
os.symlink('../foo/x', 'x', dir_fd=dir_fd)
