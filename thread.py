from threading import Lock, Thread
from time import sleep

class MyCache():
    def __init__(self):
        self.lock = Lock()
        self.data = list()

    def slow_update(self):
        more = query_database()
        self.data.append(more)

    def get_data(self):
        with self.lock:
            self.slow_update()

        return self.data

my_cache = MyCache() # at module level


def thing():
    i = 0
    for i in range(3):
        print('Thread: %d' % i)
        sleep(1)

t = Thread(target=thing)
t.start()
t.join(1.5)
print('Tired of waiting', t.is_alive())
t.join()
print('Finished', t.is_alive())
