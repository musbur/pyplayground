from logging import getLogger, Handler
import socket


def SocketHandler(Handler):
    def emit(self, record):
        print('Received a record')

_log = getLogger(__name__)

_log.addHandler(SocketHandler)

_log.setLevel(10)

_log.debug('Logging something')


def y():
    sock = socket.socket(socket.AF_UNIX)
    sock.connect("/tmp/logmail.socket")
    sock.send(b"Hello, world")
    sock.close()
