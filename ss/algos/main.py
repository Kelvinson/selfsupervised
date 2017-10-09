import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from ss.envs.ball_env import BallEnv, obs_to_goal

from ss.algos.trainer import Trainer
from ss.algos.params import get_params

from baselines.ddpg.noise import *
from ss.path import get_expdir

import gym
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import click

@click.command()
@click.option('--expname', default=None)
def cmdrun(expname, **kwargs):
    run(expname, **kwargs)

def run(expname=None, **kwargs):
    params = get_params(her=True, **kwargs)
    layer_norm = params["layer_norm"]
    evaluation = True

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
    env = BallEnv()
    params["observation_shape"] = env.observation_space.shape
    params["action_shape"] = env.action_space.shape
    params["obs_to_goal"] = env.obs_to_goal
    params["goal_idx"] = env.goal_idx
    params["reward_fn"] = env.reward_fn

    # env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank), allow_early_resets=True)
    gym.logger.setLevel(logging.WARN)

    # Configure components.
    params["env"] = env

    # Seed everything to make things reproducible.
    seed = np.random.randint(0, 1000000)
    params["seed"] = seed
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    t = Trainer(**params)
    t.train()
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

if __name__ == '__main__':
    cmdrun()
