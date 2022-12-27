import logging


_log = logging.getLogger(__name__)

logging.basicConfig(level=10)
_log.setLevel(logging.DEBUG)
_log.debug('Debug')
