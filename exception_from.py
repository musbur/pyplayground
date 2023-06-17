class FileError(Exception):
    def __init__(self, msg=None, info=None):
        self.info = info
        self.msg = msg

    def __str__(self):
        return f"{self.info}: {self.msg}"

try:
    raise FileError("Message")
except FileError as e:
    e.info = "Info"
    raise RuntimeError from e
