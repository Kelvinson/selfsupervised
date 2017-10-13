"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import os
import tempfile
import time
import click
import machine

from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")

def kill():
    for x in m.ls():
        name = x['Name']
        if name:
            cmd = "/usr/local/bin/docker-machine rm -f " + name
            Popen(cmd.split(" ")).wait()

if __name__ == "__main__":
    kill()
