""" Proximal Policy Gradient

Utilized Modules: a3c

"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from network.base import Deep_layer
from network.pg import Loss, Backpropagation

from network.a3c import ActorCritic, a3c


class PPO(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.001,
                 sess=None, global_network=None):
        """ Initialize AC network and required parameters """
        super().__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            loss=self._ppo_loss)

    def _ppo_loss(
        self,
        logits, action, reward,
        td_target, critic,
        entropy_beta=0.01, critic_beta=0,
        actor_weight=None, critic_weight=None,
        name_scope='loss'
    ):
        epsilon = 0.2
        target_pi = self.global_network.actor
        target_v = self.global_network.critic
        with tf.name_scope(name_scope):
            entropy = -tf.reduce_mean(logits * tf.log(logits), name='entropy')

            action_OH = tf.one_hot(action, self.action_size)
            ratio = tf.maximum(tf.reduce_sum(logits * action_OH, 1), 1e-13) / \
                tf.maximum(tf.reduce_sum(target_pi * action_OH, 1), 1e-13)
            ratio = tf.clip_by_value(ratio, 0, 10)
            ppo1 = reward * ratio
            ppo2 = reward * tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -tf.reduce_mean(tf.minimum(ppo1, ppo2)) - entropy_beta * entropy

            clipped_value_estimate = target_v + tf.clip_by_value(self.critic - target_v, 1 - epsilon, epsilon)
            critic_v1 = tf.squared_difference(clipped_value_estimate, td_target)
            critic_v2 = tf.squared_difference(self.critic, td_target)
            critic_loss = tf.reduce_mean(tf.maximum(critic_v1, critic_v2))

        return actor_loss, critic_loss, entropy

    def update_global(self, state_input, action, td_target, advantage, log=False):
        feed_dict = {
            self.state_input: state_input,
            self.action_: action,
            self.td_target_: td_target,
            self.advantage_: advantage,
            self.global_network.state_input: state_input
        }
        self.sess.run(self.update_ops, feed_dict)

        ops = [self.actor_loss, self.critic_loss, self.entropy]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        if log:
            raise NotImplementedError

        return aloss, closs, entropy

