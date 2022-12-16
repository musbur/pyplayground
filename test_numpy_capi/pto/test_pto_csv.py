import os
from .pto_csv import PTO_CSV

dir, _ = os.path.split(__file__)
testfile = os.path.join(dir,
    'Process Details - WLN8R0  b.11 - SPM6 - 16-22-41.csv')

def _test_it():
    pto = PTO_CSV(["abc", "def"])
    with open(testfile, 'rb') as csv:
        while True:
            buf = csv.read(100)
            if not buf:
                break
            pto.feed(buf)
    print(pto.header())
    print(pto.columns())
    print(pto.selected())
    print(pto.data())

if __name__ == '__main__':
    _test_it()
