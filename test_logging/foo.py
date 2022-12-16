
# Example 1:
# Why does the logger "mylog" require
# a call to the root logger in order to work?

import logging
import bar
import sys

logging.basicConfig(level=logging.DEBUG)

mylog = logging.getLogger('foo')
mylog.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt='*** {levelname} {message}', style='{')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
mylog.addHandler(handler)

mylog.debug('1 mylog.debug()')   # prints nothing
mylog.debug('3 mylog.debug()')   # works

# bar.logger.setLevel(logging.DEBUG)
bar.work()
