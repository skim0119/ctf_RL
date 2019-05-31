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

from utility.utils import store_args


class Loss:
    """Loss

    Build function for commonly used loss functions for Policy gradient

    The default is the 'softmax cross-entropy selection' for actor loss and 'TD-difference' for critic error

    """

    @staticmethod
    def softmax_cross_entropy_selection(policy, action, reward,
                                        td_target, critic,
                                        entropy_beta=0, critic_beta=0,
                                        actor_weight=None, critic_weight=None,
                                        name_scope='loss'):
        with tf.name_scope(name_scope):
            # Entropy
            entropy = -tf.reduce_mean(policy * Loss._log(policy), name='entropy')

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
                obj_func = Loss._log(tf.reduce_sum(policy * action_OH, 1))
                exp_v = obj_func * reward
                actor_loss = tf.reduce_mean(-exp_v, name='actor_loss')
            else:
                raise NotImplementedError

            if entropy_beta != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if critic_beta != 0:
                raise NotImplementedError
                # actor_loss += tf.stop_gradient(critic_beta * critic_loss)

        return actor_loss, critic_loss, entropy

    @staticmethod
    def _log(val):
        #return tf.debugging.check_numerics(tf.log(val),'log nan found')
        return tf.log(tf.clip_by_value(val, 1e-10, 1.0))


class Backpropagation:
    """Asynchronous training pipelines"""
    @staticmethod
    def asynch_pipeline(actor_loss, critic_loss,
                        a_vars, c_vars,
                        a_targ_vars, c_targ_vars,
                        lr_actor, lr_critic,
                        tau=None,
                        name_scope='sync',
                        return_gradient=False):
        # Sync with Global Network
        with tf.name_scope(name_scope):
            critic_optimizer = tf.train.AdamOptimizer(lr_critic)
            actor_optimizer = tf.train.AdamOptimizer(lr_actor)

            with tf.name_scope('local_grad'):
                a_grads = tf.gradients(actor_loss, a_vars)
                c_grads = tf.gradients(critic_loss, c_vars)
                if tau is not None:
                    for val in a_grads + c_grads:
                        val *= tau

            with tf.name_scope('pull'):
                pull_a_vars_op = [var.assign(value) for var, value in zip(a_vars, a_targ_vars)]
                pull_c_vars_op = [var.assign(value) for var, value in zip(c_vars, c_targ_vars)]
                pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

            with tf.name_scope('push'):
                update_a_op = actor_optimizer.apply_gradients(zip(a_grads, a_targ_vars))
                update_c_op = critic_optimizer.apply_gradients(zip(c_grads, c_targ_vars))
                update_ops = tf.group(update_a_op, update_c_op)

        if return_gradient:
            return pull_op, update_ops, a_grads+c_grads
        return pull_op, update_ops
