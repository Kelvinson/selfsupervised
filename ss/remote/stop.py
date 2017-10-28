"""Interface to docker-machine command line to provision machines"""

from subprocess import Popen, PIPE
import os
import tempfile
import time
import click
import machine

from multiprocessing import Process, Pool

m = machine.Machine(path="/usr/local/bin/docker-machine")
existing = [x['Name'] for x in m.ls() if x['Name']]

import libtmux
server = libtmux.Server()
s = server.list_sessions()[0]

@click.command()
@click.argument("cmd")
def parallel_run(cmd):
    for i, e in enumerate(existing):
        w = s.new_window(window_name=e)
        p = w.attached_pane
        p.send_keys('eval $(docker-machine env %s)' % e)
        p.send_keys('docker create -it anair17/ss /bin/bash')

        time.sleep(5) # TODO: how to get the ID of the docker create better
        container_id = p.cmd('capture-pane', '-p').stdout[-2]
        print(container_id)

        p.send_keys('docker cp . %s:/selfsupervised/' % container_id)
        p.send_keys('docker start %s' % container_id)
        p.send_keys('docker attach %s' % container_id)
        p.send_keys('')
        p.send_keys('python %s %d' % (cmd, i))

@click.command()
@click.argument("start", default=0)
@click.argument("end", default=1000)
def parallel_quit(start, end):
    for i, e in enumerate(existing):
        if not i in range(start, end):
            continue
        try:
            w = s.select_window(e)
            w.kill_window()
        except:
            print("did not kill", i, e)


@click.command()
@click.argument("start", default=0)
@click.argument("end", default=100)
def parallel_stop(start, end):
    end = min(len(existing), end)
    ids = range(start, end)
    names = [existing[i] for i in ids]
    pool = Pool(processes=len(ids))
    pool.map(stop_one, names)

def stop_one(e):
    cmd = '/usr/local/bin/docker-machine ssh %s "sudo killall python"' % e
    print(cmd)
    p = Popen(cmd, shell=True)
    stdout, stderr = p.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    print("success.")

if __name__ == "__main__":
    # parallel_run()
    # parallel_quit()
    parallel_stop()
