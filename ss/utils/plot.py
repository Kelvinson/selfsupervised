import matplotlib.pyplot as plt
import numpy as np

import json
from pprint import pprint
import tensorflow as tf

read_tb = lambda: None
import glob
import os
import itertools

cached_tbs = {}
cached_params = {}

def read_tb(eventfile):
    x = {}
    y = {}
    for summary in tf.train.summary_iterator(eventfile):
        s = summary.step
        for v in summary.summary.value:
            l = x.setdefault(v.tag, [])
            l.append(s)

            l = y.setdefault(v.tag, [])
            l.append(v.simple_value)

    r = {}
    for key in x:
        sorted_index = np.argsort(x[key])
        r[key] = (np.array(x[key])[sorted_index], np.array(y[key])[sorted_index])

    return r

def get_tb(eventfile):
    if eventfile not in cached_tbs:
        cached_tbs[eventfile] = read_tb(eventfile)
    return cached_tbs[eventfile]

def load_exps(dirname, paramlist):
    assert dirname[-1] == "/"
    exps = []
    for expname in glob.glob(dirname + '*'):
        expid = int(expname[len(dirname):])
        tbfilename = glob.glob(expname + "/tb/events.*")[0]
        t = get_tb(tbfilename)
        exps.append((t, paramlist[expid]))
    return exps

def read_params_from_output(filename, maxlines=200):
    if not filename in cached_params:
        f = open(filename, "r")
        params = {}
        for i in range(maxlines):
            l = f.readline()
            if not ":" in l:
                break
            kv = l[l.find("]")+1:]
            colon = kv.find(":")
            k, v = kv[:colon], kv[colon+1:]
            params[k.strip()] = v.strip()
        f.close()
        cached_params[filename] = params
    return cached_params[filename]

def prettify(p, key):
    """Postprocessing p[key] for printing"""
    return p[key]

def prettify_configuration(config):
    if not config:
        return ""
    s = ""
    for c in config:
        k, v = str(c[0]), str(c[1])
        x = ""
        x = k + "=" + v + ", "
        s += x
    return s[:-2]

true_fn = lambda p: True
def comparison(exps, key, vary = "expdir", f=true_fn, w="evaluator"):
    """exps is a list of directories
    key is the Y variable
    vary is the X variable
    f is a filter function on the exp parameters"""
    plt.figure(figsize=(10, 6))
    plt.title("Vary " + vary)
    plt.ylabel(key)
    lines = []
    for e, p in exps:
        if f(p):
            x, y = e[key]
            line, = plt.plot(x, y, label=prettify(p, vary))
            lines.append(line)
    plt.legend(handles=lines, bbox_to_anchor=(1.5, 0.75))

def split(exps, keys, vary = "expdir", split=[], f=true_fn, w="evaluator"):
    split_values = {}
    for s in split:
        split_values[s] = set()
    for e, p in exps:
        if f(p):
            for s in split:
                split_values[s].add(p[s])

    configurations = []
    for s in split_values:
        c = []
        for v in split_values[s]:
            c.append((s, v))
        configurations.append(c)
    for c in itertools.product(*configurations):
        f = lambda p: all([p[k] == v for k, v in c])
        for key in keys:
            comparison(exps, key, vary, f=f, w=w)
            plt.title(prettify_configuration(c) + " Vary " + vary)

def diagnose(exp, keys=[]):
    outputfile = glob.glob(exp + "/evaluator/*/output.txt")[0]
    p = read_params_from_output(outputfile)
    print(exp)
    for k in keys:
        print(k, p[k])
    comparison([exp], "worker/reward", w="worker")
    comparison([exp], "optim/loss_bc", w="optimizer")
    comparison([exp], "eval/success_rate")
    comparison([exp], "eval/reward")
