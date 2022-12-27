import click

@click.argument('a2')
@click.command()
def command2(a2):
    click.echo('A2: %s' %  a2)
