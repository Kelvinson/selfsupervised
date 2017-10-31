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
def clear():
    cached_tbs.clear()
    cached_params.clear()

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
    fs = glob.glob(dirname + '*')
    exps = []
    for expname in fs:
        expid = int(expname[len(dirname):])
        print(expid)
        for tbfilename in glob.glob(expname + "/tb/events.*"):
            t = get_tb(tbfilename)
            exps.append((expid, tbfilename, t, paramlist[expid]))
    exps = [(c, d) for a, b, c, d in sorted(exps)]
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
identity_fn = lambda x: x
def comparison(exps, key, vary = "expdir", f=true_fn, w="evaluator", smooth=identity_fn):
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
            y_smooth = smooth(y)
            x_smooth = x[:len(y_smooth)]
            line, = plt.plot(x_smooth, y_smooth, label=prettify(p, vary))
            lines.append(line)
    plt.legend(handles=lines, bbox_to_anchor=(1.5, 0.75))

def split(exps, keys, vary = "expdir", split=[], f=true_fn, w="evaluator", smooth=identity_fn):
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
        fsplit = lambda p: all([p[k] == v for k, v in c]) and f(p)
        for key in keys:
            comparison(exps, key, vary, f=fsplit, w=w, smooth=smooth)
            plt.title(prettify_configuration(c) + " Vary " + vary)

def diagnose(exp, keys=[]):
    e, p = exp
    for k in keys:
        print(k, p[k])
    comparison([exp], "worker/reward", w="worker")
    comparison([exp], "optim/loss_bc", w="optimizer")
    comparison([exp], "eval/success_rate")
    comparison([exp], "eval/reward")

def ma_filter(N):
    return lambda x: moving_average(x, N)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
