import logging
import sys


_log = logging.getLogger(__name__)
_log.addHandler(logging.StreamHandler(sys.stderr))
_log.setLevel(logging.DEBUG)
_log.debug('Debug')
