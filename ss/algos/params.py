"""Default parameters for DDPG"""

import collections
import numpy as np

def get_params(overrides={}):
    params = collections.OrderedDict()
    params["gamma"] = 0.95
    params["tau"] = 1.0
    params["batch_size"] = 128
    params["observation_range"] = (-5., 5.)
    params["action_range"] = (-1., 1.)
    params["return_range"] = (-np.inf, np.inf)
    params["critic_l2_reg"] = 0 # 1e-2
    params["actor_lr"] = 1e-3
    params["critic_lr"] = 1e-3
    params["seed"] = 0
    params["nb_epochs"] = 500  # with default settings, perform 1M steps total
    params["nb_epoch_cycles"] = 20
    params["nb_train_steps"] = 50  # per epoch cycle and MPI worker
    params["nb_eval_steps"] = 20  # per epoch cycle and MPI worker
    params["nb_rollout_steps"] = 20  # per epoch cycle and MPI worker
    params["render_eval"] = False
    params["render"] = False
    params["her"] = False
    params["buffer_size"] = 1000000
    params["noise_mu"] = 0.0
    params["noise_sigma"] = 1.0
    params["reward_scale"] = 1.0
    params["horizon"] = 20
    params["stats_sample"] = None
    params["layer_norm"] = True
    params["normalize_returns"] = False
    params["normalize_observations"] = True
    params["popart"] = False
    params["clip_norm"] = None
    params["buffer_size"] = 1e6

    # parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    # boolean_flag(parser, 'render-eval', default=False)
    # boolean_flag(parser, 'layer-norm', default=True)
    # boolean_flag(parser, 'render', default=False)
    # boolean_flag(parser, 'normalize-returns', default=False)
    # boolean_flag(parser, 'normalize-observations', default=True)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    # parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    # parser.add_argument('--actor-lr', type=float, default=1e-4)
    # parser.add_argument('--critic-lr', type=float, default=1e-3)
    # boolean_flag(parser, 'popart', default=False)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--reward-scale', type=float, default=1.)
    # parser.add_argument('--clip-norm', type=float, default=None)
    # parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    # parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    # parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    # parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    # parser.add_argument('--nb-rollout-steps', type=int, default=50)  # per epoch cycle and MPI worker
    # parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    # boolean_flag(parser, 'evaluation', default=False)
    # return vars(parser.parse_args())
    # agent = DDPG(actor=actor, critic=critic, memory=memory, observation_shape=env.observation_space.shape, action_shape=env.action_space.shape,
    #         gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    #         batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    #         actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #         reward_scale=reward_scale, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
    #         stats_sample=None)

    # clip_norm=None, reward_scale=1.):

    for key in overrides:
        params[key] = overrides[key]

    return params
