"""
a3c.py module includes basic modules and implementation of A3C for CtF environment.

Some of the methods are left unimplemented which makes the a3c module to be parent abstraction.

Script contains example A3C

TODO:
    - Include gradient and weight histograph for nan debug
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from network.base import Deep_layer
from network.pg import Loss, Backpropagation

from network.base import Tensorboard_utility as TB
from network.base import put_channels_on_grid
from network.a3c import a3c
from network.attention import non_local_nn_2d


class A3C_attention(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(A3C_attention, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def run_network(self, states):
        actions = []
        critics = []
        for idx, state in enumerate(states):
            feed_dict = {
                self.state_input: state[np.newaxis,:]
            }
            ops = [self.actor, self.critic]
            prob, critic = self.sess.run(ops, feed_dict)
            action = np.random.choice(self.action_size, p=prob[0] / sum(prob[0])) 
            actions.append(action)
            critics.append(critic[0])
        return actions, critics

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name):
            grid = put_channels_on_grid(net[0], -1, 8)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input')

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            net = tf.stack([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn(net, 16, pool=True, name='non_local', summary_adder=add_image)
            add_image(net, 'attention')

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            net= layers.fully_connected(
                net, 128,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[32, 64, 64],
                kernels=[5, 3, 2],
                pools=[2, 2, 2],
                flatten=True
            )

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)

        return logits, actor, critic, a_vars, c_vars
