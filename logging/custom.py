import logging

class MyLogger(logging.getLoggerClass()):
    pass

mylog = MyLogger("mytest")
log = logging.getLogger("test")
root = logging.getLogger()

h = logging.StreamHandler()
f = logging.Formatter(fmt="[%(levelname)s] %(message)s")
h.setFormatter(f)
root.addHandler(h)

for h in root.handlers:
    mylog.addHandler(h)

log.error("Normal log")
mylog.error("Test log")
