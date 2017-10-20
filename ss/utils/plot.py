import matplotlib.pyplot as plt
import numpy as np

import json
from pprint import pprint

read_tb = lambda: None
import glob
import itertools

cached_tbs = {}
cached_params = {}

def get_tb(eventfile):
    if eventfile not in cached_tbs:
        cached_tbs[eventfile] = read_tb(eventfile)
    return cached_tbs[eventfile]

def get_series(tb, key):
    s = tb[key]
    idx = ~np.isnan(s)
    x = np.arange(len(idx))[idx]
    y = s[idx]
    return x, y

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
    if key == "attention":
        a = eval(p[key])
        if a:
            return str(a[1])
    return p[key]

def prettify_configuration(config):
    if not config:
        return ""
    s = ""
    for c in config:
        k, v = c[0], c[1]
        x = ""
        if k == "attention":
            a = eval(v)
            if a:
                v = str(a[1])
            else:
                v = str(a)
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
    for e in exps:
        name = e[e.rfind('/')+1:]
        eventfile = glob.glob(e + "/%s/*/events*" % w)[0]
        outputfile = glob.glob(e + "/%s/*/output.txt" % w)[0]
        p = read_params_from_output(outputfile)
        p["expdir"] = name
        if f(p):
            tb = get_tb(eventfile)
            x, y = get_series(tb, key)
            line, = plt.plot(x, y, label=prettify(p, vary))
            lines.append(line)
    plt.legend(handles=lines, bbox_to_anchor=(1.5, 0.75))

def split(exps, keys, vary = "expdir", split=[], f=true_fn, w="evaluator"):
    split_values = {}
    for s in split:
        split_values[s] = set()
    for e in exps:
        outputfile = glob.glob(e + "/%s/*/output.txt" % w)[0]
        p = read_params_from_output(outputfile)
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
