"""Handles running experiments"""

import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from ss.algos.trainer import Trainer
from ss.algos.params import get_params

from baselines.ddpg.noise import *
from ss.path import get_expdir

import gym
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import click
import pdb

from multiprocessing import Process, Pool

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def cmdrun(args):
    """Hacky command line parsing to pass through all parameters to override"""
    kwargs = {}
    for k, v in zip(args[::2], args[1::2]):
        kwargs[k] = eval(v)
    params = get_params(**kwargs)
    run_parallel([params])

def run_parallel(paramlist, parallel=0):
    """Runs one experiment per experiment in paramlist.
    parallel is the number of experiments to run in parallel at once.
    parallel < 1 means run everything in parallel.
    """
    if len(paramlist) == 1:
        params = paramlist[0]
        run(params) # no need for any new processes
    elif len(paramlist) > 0:
        if parallel < 1:
            parallel = len(paramlist)
        pool = Pool(processes=parallel)
        pool.map(run, paramlist)

def run(params):
    expname = params["expname"]
    if not expname:
        expname = "test_" + time.strftime("%d-%m-%Y_%H-%M-%S")

    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    else:
        logdir = get_expdir(expname + "/" )
        logger.configure(logdir, ['stdout', 'log', 'json', 'tensorboard'])

    # Create envs.
    env = params["env_type"]()
    params["observation_shape"] = env.observation_space.shape
    params["action_shape"] = env.action_space.shape
    params["obs_to_goal"] = env.obs_to_goal
    params["goal_idx"] = env.goal_idx
    params["reward_fn"] = env.reward_fn

    # env = gym.make(env_id)
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank), allow_early_resets=True)
    gym.logger.setLevel(logging.WARN)

    # Configure components.
    params["env"] = env

    # Seed everything to make things reproducible.
    seed = params["seed"]
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    if params["trainer"]:
        tr = params["trainer"]
    else:
        tr = Trainer
    t = tr(**params)
    t.train()
    env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

if __name__ == '__main__':
    cmdrun()
