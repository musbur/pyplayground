import threading
from time import sleep



lock = threading.Lock()
def test_lock():
    lock.acquire()
    print(threading.get_ident(), 'Locked')
    sleep(1)
    lock.release()
    print(threading.get_ident(), 'Released')

for i in range(3):
    t = threading.Thread(target=test_lock)
    t.start()
