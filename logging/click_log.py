import logging
import sys
import click

_log = logging.getLogger(__name__)

@click.group()
def cli():
    _log.addHandler(logging.StreamHandler(sys.stderr))
    _log.setLevel(logging.DEBUG)
    _log.debug('From Group')

@cli.command()
def log():
    _log.debug('From Command')

if __name__ == '__main__':
    cli()
