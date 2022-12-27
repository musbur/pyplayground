from importlib import import_module
import click

def get_commands(module_name):
    commands = dict()
    mod = import_module(module_name)
    for e in dir(mod):
        cmd = getattr(mod, e)
        if isinstance(cmd, click.Command):
            commands[e] = cmd

    class X(click.MultiCommand):

        def list_commands(self, ctx):
            return commands.keys()

        def get_command(self, ctx, name):
            return commands[name]

    return X


module_commands = {n: get_commands(n) for n in ['sub1', 'sub2']}

class M(click.MultiCommand):

    def list_commands(self, ctx):
        return module_commands.keys()

    def get_command(self, ctx, name):
        return module_commands[name]()

@click.command(cls=M)
@click.option('-o', '--optio', is_flag=True)
def all(**options):
    print(options)
    pass

if __name__ == '__main__':
    all()

