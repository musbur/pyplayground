import os
from socketserver import UnixStreamServer, StreamRequestHandler

SOCKET = '/tmp/test.socket'

class Handler(StreamRequestHandler):

    def handle(self):
        data = self.rfile.read()
        print(data)

if __name__ == '__main__':
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)
    with UnixStreamServer(SOCKET, Handler) as server:
        server.serve_forever()
