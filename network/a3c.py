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


class a3c:
    """ A3C Module

    Base module for a3c without any variation.
    """

    @store_args
    def __init__(
        self,
        in_size,
        action_size,
        scope,
        lr_actor=1e-4,
        lr_critic=1e-4,
        entropy_beta=0.001,
        sess=None,
        global_network=None,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."
        if global_network is None: # For primary graph, pipe to self
            global_network = self

        with self.sess.as_default(), self.sess.graph.as_default():
            loss = kwargs.get('loss', Loss.softmax_cross_entropy_selection)
            backprop = kwargs.get('back_prop', Backpropagation.asynch_pipeline)

            with tf.variable_scope(scope):
                self._build_placeholder(in_size)

                # get the parameters of actor and critic networks
                self.actor, self.critic, self.a_vars, self.c_vars = self._build_network(self.state_input)

                # Local Network
                train_args = (self.action_, self.advantage_, self.td_target_)
                loss = loss(self.actor, *train_args, self.critic)
                self.actor_loss, self.critic_loss, self.entropy = loss

                self.pull_op, self.update_ops = backprop(
                    self.actor_loss, self.critic_loss,
                    self.a_vars, self.c_vars,
                    global_network.a_vars, global_network.c_vars,
                    lr_actor, lr_critic
                )

            self.merged_summary_op = self._build_summary(self.a_vars + self.c_vars)

    def _build_placeholder(self, input_shape):
        self.state_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name='state')
        self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
        self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
        self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')

    def _build_summary(self, vars_list: list):
        """ _build_summary (Implementation required)

        Given the list of tensor variables, returns summary operator of histogram.
        Mostly used for summarizing the weights of the layer.

        ex)
            var_record = _build_summary(actor_vars)
            sess.run(var_record, feed_dict)

        Parameters
        ----------------
        vars_list: variables in List form

        Returns
        ----------------
        Scalar Tensor of String: Serialized Summary protocol for variables histogram
        """

        summaries = []
        for var in vars_list:
            var_name = var.name.replace(":", "_")  # colon (:) is not allowed in TF board
            summaries.append(tf.summary.histogram(var_name, var))
        return tf.summary.merge(summaries)

    def _build_grad_summary(self, vars_list: list, grads_list: list):
        raise NotImplementedError

    def _build_network(self, input_holder):
        """ _build_network

        Network is not pre-implemented for further customization.

        Parameters
        ----------------
        input_holder : [tf.placeholder]

        Returns
        ----------------
        actor : [Tensor]
        critic: [Tensor]
        a_vars : [List]
            variables in Actor layers
        c_vars : [List]
            variables in Critic layers

        """
        raise NotImplementedError

    def run_network(self, states):
        """ run_network
        Parameters
        ----------------
        states : [List/np.array]

        Returns
        ----------------
            action : [List]
            critic : [List]
        """

        a_probs, critics = self.run_sample(states)
        actions = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs]
        return actions, critics

    def run_sample(self, states):
        feed_dict = {self.state_input: states}
        return self.sess.run([self.actor, self.critic], feed_dict)

    def update_global(self, state_input, action, td_target, advantage, log=False):
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
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage}
        self.sess.run(self.update_ops, feed_dict)

        ops = [self.actor_loss, self.critic_loss, self.entropy]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        if log:
            raise NotImplementedError

        return aloss, closs, entropy

    def pull_global(self):
        self.sess.run(self.pull_op)

    def initialize_vars(self):
        var_list = self.get_vars
        init = tf.initializers.variables(var_list)
        self.sess.run(init)

    @property
    def get_vars(self):
        return self.a_vars + self.c_vars


class ActorCritic(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.001,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(ActorCritic, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        with tf.variable_scope('actor'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[32, 64, 64],
                kernels=[5, 3, 2],
                pools=[2, 2, 2],
                flatten=True
            )
            net = layers.fully_connected(net, 128)
            actor = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=tf.nn.softmax)

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

        return actor, critic, a_vars, c_vars
