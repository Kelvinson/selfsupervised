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

def parallel_quit():
    for i, e in enumerate(existing):
        w = s.select_window(e)
        w.kill_window()

if __name__ == "__main__":
    # parallel_run()
    parallel_quit()
