""" Policy Gradient Module

This module contains classes and definition to assist building policy graient model.

Fuctions:
    build_loss (list:Tensor, list:Tensor, list:Tensor):
        Returns the actor loss, critic loss, and entropy.

Todo:
    * Try to use tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    * Try to create general policy gradient module
    * Autopep8
    * Docstrings

"""

import tensorflow as tf

import numpy as np

from pg import _log


class Loss:
    """Loss

    Build function for commonly used loss functions for Policy gradient

    The default is the 'softmax cross-entropy selection' for actor loss and 'TD-difference' for critic error

    """

    @staticmethod
    def ppo(policy, log_prob, old_log_prob,
            action, advantage,
            td_target, critic,
            entropy_beta=0, critic_beta=0,
            actor_weight=None, critic_weight=None,
            name_scope='loss'):
        with tf.name_scope(name_scope):
            # Entropy
            entropy = -tf.reduce_mean(policy * _log(policy), name='entropy')

            # Critic Loss
            if critic_weight is None:
                td_error = td_target - critic
                critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
            else:
                raise NotImplementedError

            # Actor Loss
            if actor_weight is None:
                action_size = tf.shape(policy)[1]
                action_OH = tf.one_hot(action, action_size, dtype=tf.float32)
                obj_func = _log(tf.reduce_sum(policy * action_OH, 1))
                exp_v = obj_func * advantage
                actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')
            else:
                raise NotImplementedError

            if entropy_beta != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if critic_beta != 0:
                raise NotImplementedError
                # actor_loss += tf.stop_gradient(critic_beta * critic_loss)

        return actor_loss, critic_loss, entropy

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

    @staticmethod
    def _log(val):
        #return tf.debugging.check_numerics(tf.log(val),'log nan found')
        return tf.log(tf.clip_by_value(val, 1e-10, 1.0))
