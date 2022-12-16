from numpy_capi import create_ndarray
from numpy_capi_class import PTO_CSV

if __name__ == '__main__':
    obj = create_ndarray(2, 5)
    print(obj)
    pto = PTO_CSV()
    pto.data()
