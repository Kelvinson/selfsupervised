"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import os
import tempfile
import time
import click
import machine

from ss.path import DOCKER_IMAGE_NAME
from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")
existing = [x['Name'] for x in m.ls()]

@click.command()
@click.argument("start", default=0)
@click.argument("end", default=1)
@click.argument("prefix", default="ss")
def create(start, end, prefix):
    ps = []
    pool = Pool(processes=int(end-start))
    names = [prefix + str(i) for i in range(start, end)]
    pool.map(create_one, names)

def create_one(name):
    if name in existing:
        print(name, "already exists.")
        return
    # CREATE = """create --driver amazonec2 --amazonec2-region us-west-2 --amazonec2-request-spot-instance --amazonec2-spot-price 0.1 --amazonec2-instance-type m4.large"""
    CREATE = """create --driver amazonec2 --amazonec2-region us-east-1 --amazonec2-request-spot-instance --amazonec2-spot-price 0.1 --amazonec2-instance-type m4.large"""
    cmd = ["/usr/local/bin/docker-machine"] + CREATE.split(" ") + [name]
    p = Popen(cmd).wait()
    if p:
        print(name, "HALTING DUE TO ERROR.")
        return

    docker_pull(name)

def docker_pull(machine_name):
    set_env(machine_name)
    cmd = os.path.expandvars("docker login --username=$DOCKER_USER --password=$DOCKER_PASS")
    cmd_list = cmd.split(" ")
    p = Popen(cmd_list).wait()
    print(machine_name, "login", p)

    cmd = "docker pull " + DOCKER_IMAGE_NAME
    cmd_list = cmd.split(" ")
    p = Popen(cmd_list).wait()
    print(machine_name, "docker pull", p)

def set_env(machine_name):
    e = m.env(machine_name)
    e = e[1:e.index("#"):2]
    for x in e:
        k, v = x.split("=")
        os.environ[k] = eval(v)

if __name__ == "__main__":
    create()
