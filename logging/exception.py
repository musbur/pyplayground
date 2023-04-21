import logging
from pprint import pprint
import pdb

def log_something():
    log.error("Normal error: %d", 5)
    try:
        raise RuntimeError("Exception!")
    except Exception as e:
        log.exception(e)

class FmtExcOneliner(logging.Formatter):

    def format(self, rec):
        # ---- copied from logging.__init__
        rec.message = rec.getMessage()
        if self.usesTime():
            rec.asctime = self.formatTime(rec, self.datefmt)
        s = self.formatMessage(rec)
        # ----
        if rec.exc_info:
           return ("{s} (@{x.co_filename}:{x.co_firstlineno})"
                   .format(s=s, x=rec.exc_info[2].tb_frame.f_code))
        else:
            return s

s_handler = logging.StreamHandler()
s_handler.setFormatter(FmtExcOneliner(
    fmt="Stream: %(levelname)s %(asctime)s %(message)s"))
f_handler = logging.FileHandler("log.txt", "w")
f_handler.setFormatter(logging.Formatter(fmt="File: %(message)s"))

log = logging.getLogger()
log.addHandler(f_handler)
log.addHandler(s_handler)


log_something()
