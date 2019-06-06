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

from network.pg import Loss as pgLoss


class Loss:
    """Loss

    Build function for commonly used loss functions for Policy gradient

    The default is the 'softmax cross-entropy selection' for actor loss and 'TD-difference' for critic error

    """

    @staticmethod
    def ppo(policy, log_prob, old_log_prob,
            action, advantage,
            td_target, critic,
            entropy_beta=0.001, critic_beta=0,
            actor_weight=None, critic_weight=None,
            eps=0.2,
            name_scope='loss'):
        with tf.name_scope(name_scope):
            # Entropy
            entropy = -tf.reduce_mean(policy * pgLoss._log(policy), name='entropy')

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

                log_prob = tf.reduce_sum(log_prob * action_OH, 1)
                #old_log_prob = tf.reduce_sum(old_log_prob * action_OH, 1)
                # Clipped surrogate function
                ratio = log_prob / old_log_prob
                surrogate = ratio * advantage
                clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
                actor_loss = tf.reduce_mean(-tf.minimum(surrogate, advantage), name='actor_loss')
            else:
                raise NotImplementedError

            if entropy_beta != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if critic_beta != 0:
                raise NotImplementedError
                # actor_loss += tf.stop_gradient(critic_beta * critic_loss)

        return actor_loss, critic_loss, entropy
