import time
from datetime import datetime, timedelta
import sched


scheduler = sched.scheduler(time.time, time.sleep)

def task():
    print(datetime.now())
    next_ts = datetime.now() + timedelta(minutes=0, seconds=10)
    print('Scheduling next for %s\n' % next_ts)
    scheduler.enterabs(next_ts.timestamp(), 1, task)


scheduler.enterabs(1, 1, task)
scheduler.run()
