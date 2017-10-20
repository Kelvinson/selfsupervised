"""Default parameters for DDPG"""

import collections
import numpy as np

from ss.envs.ball_env import BallEnv

def get_params(**kwargs):
    params = collections.OrderedDict()
    params["expname"] = None
    params["gamma"] = 0.95
    params["tau"] = 1.0
    params["batch_size"] = 128
    params["observation_range"] = (-5., 5.)
    params["action_range"] = (-1., 1.)
    params["return_range"] = (-20.0, 0.0)
    params["critic_l2_reg"] = 0 # 1e-2
    params["actor_lr"] = 1e-3
    params["critic_lr"] = 1e-3
    params["seed"] = 0
    params["nb_epochs"] = 5000
    params["nb_epoch_cycles"] = 20
    params["nb_train_steps"] = 50  # per epoch cycle and MPI worker
    params["render"] = False
    params["her"] = False
    params["buffer_size"] = 1000000
    params["noise_mu"] = 0.0
    params["noise_sigma"] = 0.1
    params["reward_scale"] = 1.0
    params["horizon"] = 20
    params["stats_sample"] = None
    params["layer_norm"] = True
    params["normalize_returns"] = False
    params["normalize_observations"] = True
    params["popart"] = False
    params["clip_norm"] = None
    params["her"] = False
    params["env_type"] = BallEnv

    for key in kwargs:
        params[key] = kwargs[key]

    return params
