import os
from .pto_csv import PTO_CSV

dir, _ = os.path.split(__file__)

CORRECT = 'Process Details_correct.csv'
INCORRECT = 'Process Details_incorrect.csv'

def test_pandas():
    import pandas
    pto = PTO_CSV(("TIME", "ActualPower", "ActualVoltage",
                          "ESCControl SV_actualCapacity"))
    csv_file = os.path.join(dir, CORRECT)
    with open(csv_file, 'rb') as csv:
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


def test_full():
    pto = PTO_CSV(['ME'])
    csv_file = os.path.join(dir, INCORRECT)
    with open(csv_file, 'rb') as csv:
        while True:
            buf = csv.read(100)
            if not buf:
                break
            pto.feed(buf)
    print(pto.selected())

if __name__ == '__main__':
    test_full()
