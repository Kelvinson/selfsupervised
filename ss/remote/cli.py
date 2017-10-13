"""Interface to docker-machine command line to provision and use machines"""

import machine
from subprocess import Popen, PIPE
import os
import tempfile
import time
import click

m = machine.Machine(path="/usr/local/bin/docker-machine")

@click.command()
def ls():
    for x in m.ls():
        print(x)
        print(m.ip(x['Name']))

if __name__ == "__main__":
    ls()
