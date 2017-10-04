"""Implementation of DDPG adapted from OpenAI Baselines"""

from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from rllab.misc import logger

import ss.algos.network as network
from ss.algos.replay_buffer import ReplayBuffer

from baselines import logger
from baselines.common.mpi_adam import MpiAdam
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ddpg.util import reduce_std, mpi_mean
import pdb

class DDPG(object):
    def __init__(self, **params):
        for k in params:
            setattr(self, k, params[k])

        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape, name='obs1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        # TODO: normalization, preferably with minimal code (use TF instead)
        self.pi_tf = self.build_pi(self.obs0)
        self.Q1_tf = self.build_Q(self.obs0, self.actions, "Q") # real actions
        self.Q2_tf = self.build_Q(self.obs0, self.pi_tf, "Q", True) # pi actions for backprop

        # target critic
        self.pi_obs1_tf = self.build_pi(self.obs1, reuse=True)
        self.Q_target_tf = self.build_Q(self.obs1, self.pi_obs1_tf, "Q_target")

        # create update function
        q_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Q')
        target_q_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Q_target')
        update_target_fn = []
        for var, var_target in zip(sorted(q_vars,        key=lambda v: v.name),
                                   sorted(target_q_vars, key=lambda v: v.name)):
            var_polyak = (1. - self.tau) * var_target + self.tau * var
            update_target_fn.append(var_target.assign(var_polyak))
        self.update_target_fn = tf.group(*update_target_fn)

        self.setup_replay_buffer()
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()

    def build_pi(self, o, reuse=False):
        output = network.build_mlp("actor", o, [64, 64, self.action_shape[0]], reuse=reuse)
        return tf.tanh(output) # and scale to min/max action range

    def build_Q(self, o, u, name, reuse=False):
        input_Q_tf = tf.concat([o, u], axis=1)
        Q_tf = network.build_mlp(name, input_Q_tf, [64, 64, 1], reuse=reuse)
        return tf.clip_by_value(Q_tf, self.return_range[0], self.return_range[1])

    def setup_replay_buffer(self):
        if self.her:
            assert False
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.action_shape, self.observation_shape)

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.Q2_tf)
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor")
        actor_shapes = [var.get_shape().as_list() for var in actor_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, actor_vars) # , clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=actor_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q")
        self.critic_loss = tf.reduce_mean(tf.square(self.Q1_tf - self.critic_target))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in critic_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in critic_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, critic_vars) # , clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=critic_vars, beta1=0.9, beta2=0.999, epsilon=1e-08)

    def pi(self, obs, apply_noise=True, compute_Q=True):
        actor_tf = self.pi_tf
        feed_dict = {self.obs0: [obs]}
        action, q = self.sess.run([actor_tf, self.Q2_tf], feed_dict=feed_dict)
        # action = np.clip(action, self.action_range[0], self.action_range[1]) # assert
        if apply_noise:
            action = action + np.random.normal(self.noise_mu, self.noise_sigma)
        return action, q

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1)

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        target_Q = self.sess.run(self.Q_target_tf, feed_dict={
            self.obs1: batch['obs1'],
            self.rewards: batch['rewards'],
        })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.update_target_fn)

    def update_target_net(self):
        self.sess.run(self.update_target_fn)

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.her:
            self.memory.flush()
