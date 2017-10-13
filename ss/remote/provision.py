"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import os
import tempfile
import time
import click
import machine

from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")

def create(prefix, n):
    ps = []
    pool = Pool(processes=n)
    names = [prefix + str(i) for i in range(n)]
    pool.map(create_one, names)

def create_one(name):
    CREATE = """create --driver amazonec2 --amazonec2-region us-west-2 --amazonec2-request-spot-instance --amazonec2-spot-price 0.1 --amazonec2-instance-type m3.medium"""
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

    cmd = "docker pull anair17/ss"
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
    create("abcde", 2)
