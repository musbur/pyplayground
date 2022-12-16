import os
import pandas
from .pto_csv import PTO_CSV

dir, _ = os.path.split(__file__)
testfile = os.path.join(dir,
    'Process Details - WLN8R0  b.11 - SPM6 - 16-22-41.csv')

def _test_it():
    pto = PTO_CSV(("TIME", "ActualPower", "ActualVoltage",
                          "ESCControl SV_actualCapacity"))
    with open(testfile, 'rb') as csv:
        while True:
            buf = csv.read(100)
            if not buf:
                break
            pto.feed(buf)
    print(pto.selected())
    df = pandas.DataFrame(pto.data(), columns=pto.selected())
    df['Time'] = pandas.to_datetime(df.TIME, unit="s")
    df.TIME = df.TIME - df.TIME[0]
    print(df.loc[df.iloc[:,3] > 0])

if __name__ == '__main__':
    _test_it()
