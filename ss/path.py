import os

is_docker = os.path.isfile("/.dockerenv")

if is_docker:
    PROJECTDIR = "/selfsupervised/"
    DATADIR = "/selfsupervised/data/"
    EXPDIR = "/selfsupervised/data/experiments/"
else:
    PROJECTDIR = "/Users/ashvin/code/selfsupervised/"
    DATADIR = "/Users/ashvin/code/ssdata/"
    EXPDIR = "/Users/ashvin/code/ssdata/s3/experiments/"

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_expdir(name):
    return EXPDIR + name
