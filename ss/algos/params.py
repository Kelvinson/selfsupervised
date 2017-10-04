"""Default parameters for DDPG"""

import collections
import numpy as np

def get_params(overrides={}):
    params = collections.OrderedDict()
    params["gamma"] = 0.99
    params["tau"] = 0.001
    params["batch_size"] = 128
    params["observation_range"] = (-5., 5.)
    params["action_range"] = (-1., 1.)
    params["return_range"] = (-np.inf, np.inf)
    params["critic_l2_reg"] = 0.
    params["actor_lr"] = 1e-4
    params["critic_lr"] = 1e-3
    params["seed"] = 0
    params["nb_epochs"] = 500  # with default settings, perform 1M steps total
    params["nb_epoch_cycles"] = 20
    params["nb_train_steps"] = 50  # per epoch cycle and MPI worker
    params["nb_eval_steps"] = 100  # per epoch cycle and MPI worker
    params["nb_rollout_steps"] = 100  # per epoch cycle and MPI worker
    params["render_eval"] = False
    params["render"] = False
    params["her"] = False
    params["buffer_size"] = 1000000
    params["noise_mu"] = 0.0
    params["noise_sigma"] = 1.0
    params["reward_scale"] = 1.0
    params["horizon"] = 20

    # clip_norm=None, reward_scale=1.):

    for key in overrides:
        params[key] = overrides[key]

    return params
