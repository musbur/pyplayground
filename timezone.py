import datetime
import pytz

mytz = pytz.timezone("europe/berlin")
print(mytz)

dt = datetime.datetime(2023, 3, 26, 0, 59, 59, tzinfo=pytz.utc)
print(dt)
dt1 = dt.astimezone(mytz)
print(dt1)
dt2 = dt1 + datetime.timedelta(seconds=1)
print(dt2)
dt3 = mytz.normalize(dt2)
print(dt3)
dtx = datetime.datetime(2023, 4, 26, 0, 59, 59)
print(type(mytz.dst(dtx)))
