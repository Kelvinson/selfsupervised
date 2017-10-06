"""Copied from CS 294"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, batch_norm
import numpy as np
import pdb

xavier = xavier_initializer(
    uniform=True,
    seed=None,
    dtype=tf.float32
)

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x, size, name, weight_init=xavier):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w) + b

def build_mlp(
        scope,
        input_tensor,
        layers,
        activation=tf.nn.relu,
        output_activation=None,
        reuse=False,
        ):
    with tf.variable_scope(scope, reuse=reuse):
        # cur = batch_norm(input_tensor)
        cur = input_tensor
        for i, size in enumerate(layers[:-1]):
            cur = dense(cur, size, str(i), normc_initializer())
            cur = activation(cur)
        size = layers[-1]
        cur = dense(cur, size, "output", normc_initializer())
        if output_activation:
            cur = output_activation(cur)
    return cur

if __name__ == "__main__":
    pass
