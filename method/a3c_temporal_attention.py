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
from network.a3c import a3c
from network.attention import non_local_nn


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

    def update_global(self, state_input, action, td_target, advantage):
        """ update_global

        Run all update and back-propagation sequence given the necessary inputs.

        Parameters
        ----------------
        log : [bool]
             logging option

        Returns
        ----------------
        aloss : [Double]
        closs : [Double]
        entropy : [Double]
        """
        # Update main network
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage}
        self.sess.run(self.update_ops, feed_dict)

        ops = [self.actor_loss, self.critic_loss, self.entropy]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        return aloss, closs, entropy

    def log_all(self, writer, state_input, action, td_target, advantage, global_episodes):
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage}
        ops = [self.merged_grad_summary_op, self.merged_summary_op]
        grad_sum, val_sum = self.sess.run(ops, feed_dict)

        writer.add_summary(grad_sum, global_episodes)
        writer.add_summary(val_sum, global_episodes)

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        with tf.variable_scope('actor'):
            net = input_hold
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=32, kernel_size=(3,3,1))
            net = non_local_nn(net, 16, pool=True, name='non_local')

            num_batch, w, t, h, ch = net.get_shape().as_list()  # h as sequence
            net = tf.reshape(net, [-1, w, t, h*ch])

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=32, stride = 2, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=32, kernel_size=2)
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold[:,:,:,-1,:],
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

        return logits, actor, critic, a_vars, c_vars
