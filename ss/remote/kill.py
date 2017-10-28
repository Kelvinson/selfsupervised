"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import os
import tempfile
import time
import click
import machine

from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")

@click.command()
@click.argument("prefix", default="ss")
@click.argument("start", default=0)
@click.argument("end", default=1000)
def kill(prefix, start, end):
    N = len(prefix)
    R = set(range(start, end))
    input(("kill", R, "?"))
    for x in m.ls():
        name = x['Name']
        if name and name[:N] == prefix:
            try:
                i = int(name[N:])
                if i in R:
                    cmd = "/usr/local/bin/docker-machine rm -f " + name
                    Popen(cmd.split(" ")).wait()
            except:
                print("Skipping", name)

if __name__ == "__main__":
    kill()
