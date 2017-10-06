import os

PROJECTDIR = "/Users/ashvin/code/selfsupervised/"
DATADIR = "/Users/ashvin/code/ssdata/"

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_expdir(name):
    return DATADIR + "experiments/" + name
