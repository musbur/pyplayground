"""
import os
from .read_times import read_times

d, _ = os.path.split(__file__)
test_file = os.path.join(d, '')

start, end = read_times(test_file)

print(start, end)
"""
