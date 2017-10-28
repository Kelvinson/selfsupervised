import shelve
import machine

def docker_create(machine_name):
    set_env(machine_name)
    cmd = 'docker create -it anair17/mj13 /bin/bash'
    cmd_list = cmd.split(" ")
    p = Popen(cmd_list).wait()
    print(machine_name, "docker create", p)

if __name__ == "__main__":
    m = machine.Machine(path="/usr/local/bin/docker-machine")
    existing = [x['Name'] for x in m.ls()]

    d = shelve.open("machines")
    m = d["containers"]
    c = d["machines"]

    for e in existing:
        docker_create(
    d["machines"] = []
    d["containers"] = []
    d.close()
