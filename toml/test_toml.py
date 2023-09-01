from pprint import pprint
import tomli

with open("dham.toml", "rt") as fh:
    config = tomli.load(fh)
pprint(config)
