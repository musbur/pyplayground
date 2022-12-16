import logging

logger = logging.getLogger('bar')

logger.addHandler(logging.NullHandler())

def work():
    logger.error('error()')
