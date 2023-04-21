from click.testing import CliRunner
from cli import run

def test_run():
    runner = CliRunner()
    result = runner.invoke(run, [])
