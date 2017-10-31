"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import subprocess
import os
import tempfile
import time
import click
import machine
from ss.path import DOCKER_IMAGE_NAME

from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")
existing = [x['Name'] for x in m.ls() if x['Name']]

import libtmux
server = libtmux.Server()
s = server.list_sessions()[0]

def set_env(machine_name):
    e = m.env(machine_name)
    e = e[1:e.index("#"):2]
    for x in e:
        k, v = x.split("=")
        os.environ[k] = eval(v)

def docker_create(machine_name):
    set_env(machine_name)
    # cmd = 'docker create -it anair17/mj13 /bin/bash'
    cmd = 'docker create -it %s /bin/bash' % DOCKER_IMAGE_NAME
    cmd_list = cmd.split(" ")
    p = Popen(cmd_list, stdout=subprocess.PIPE).communicate()
    docker_id = p[0].strip()
    return docker_id.decode('UTF-8')

@click.command()
@click.argument("cmd")
@click.argument("start", default=0)
@click.argument("end", default=1000)
def parallel_run(cmd, start, end):
    machines = [e for i, e in enumerate(existing) if i in range(start, end)]
    pool = Pool(processes=len(machines))
    ids = pool.map(docker_create, machines)

    for i, e in enumerate(machines):
        if not i in range(start, end):
            continue
        w = s.new_window(window_name=e)
        p = w.attached_pane
        p.send_keys('eval $(docker-machine env %s)' % e)
        container_id = ids[i]
        print(i, e, container_id)

        # p.send_keys('docker cp . %s:/selfsupervised/' % container_id)
        p.send_keys('docker cp ss %s:/selfsupervised/' % container_id)
        p.send_keys('docker start %s' % container_id)
        p.send_keys('docker attach %s' % container_id)
        p.send_keys('')
        p.send_keys('python %s %d' % (cmd, i - start))

def parallel_quit():
    for i, e in enumerate(existing):
        w = s.select_window(e)
        w.kill_window()

if __name__ == "__main__":
    parallel_run()
    # parallel_quit()
