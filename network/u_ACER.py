import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import random

from utility.utils import store_args

from network.base import Deep_layer, Custom_initializers


class UACER:
    """UACER
    Universal Actor-Critic with Experience Replay
    Built for HER + ACER implementation for CtF Environment
    """

    @store_args
    def __init__(self, in_size, gps_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4, grad_clip_norm=0,
                 entropy_beta=0.001, retrace_lambda=0.202,
                 sess=None, global_network=None, set_global=False):
        """ Initialize AC network and required parameters """

        with tf.variable_scope(scope):
            # Build actor and critic network weights.
            with tf.name_scope('state'):
                self.local_state_ = tf.placeholder(shape=in_size, dtype=tf.float32, name='local_state')
                self.gps_state_ = tf.placeholder(shape=gps_size, dtype=tf.float32, name='gps_state')
                self.goal_state_ = tf.placeholder(shape=gps_size, dtype=tf.float32, name='goal')

            self.actor, self.critic, self.a_vars, self.c_vars = self.build_network(self.local_state_, self.gps_state_, self.goal_state_)

            if set_global:
                # Global Network
                # Optimizer
                self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor)
                self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic)
            else:
                # Local Network
                # Gradient and Pipeline
                with tf.name_scope('backprop_pipeline'):
                    self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                    self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                    self.adv_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')
                    self.actor_correction_ = tf.placeholder(shape=[None], dtype=tf.float32, name='retrace_holder')
                    self.critic_correction_ = tf.placeholder(shape=[None], dtype=tf.float32, name='retrace_prod_holder')

                self.actor_loss, self.critic_loss, self.entropy = self.build_loss()
                self.a_grads, self.c_grads = self.build_gradient()
                self.pull_a_vars, self.pull_c_vars, self.update_a, self.update_c = self.build_pipeline()

    def build_network(self, input_, gps_, goal_):
        with tf.variable_scope('actor'):
            state_array = Deep_layer.conv2d_pool(input_layer=input_,
                                                 channels=[32, 64, 64],
                                                 kernels=[5, 3, 2],
                                                 pools=[2, 2, 1],
                                                 strides=[2, 2, 1],
                                                 padding='VALID',
                                                 flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64, 64, 64],
                                      dropout=1.0)
            goal_array = Deep_layer.fc(input_layer=goal_,
                                       hidden_layers=[64, 64, 64],
                                       dropout=1.0,
                                       reuse=True)

            net = tf.concat([state_array, gps_array, goal_array], 1)

            actor = layers.fully_connected(net, self.action_size,
                                           activation_fn=tf.nn.softmax,
                                           scope='actor')

        with tf.variable_scope('critic'):
            critic = layers.fully_connected(net, 1,
                                            activation_fn=None)
            critic = tf.reshape(critic, (-1,))

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')

        return actor, critic, a_vars, c_vars

    def build_loss(self):
        # Actor Loss
        with tf.name_scope('actor_train'):
            entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
            action_OH = tf.one_hot(self.action_, self.action_size)
            pi_t = tf.reduce_sum(self.actor * action_OH, 1)  # policy for corresponding state and action

            #actor_correction = self.retrace_lambda * tf.minimum(1.0, self.actor_correction_)  # HP
            actor_correction = self.actor_correction_
            exp_v = tf.log(pi_t) * self.adv_ * actor_correction + self.entropy_beta * entropy
            actor_loss = -tf.reduce_mean(exp_v, name='actor_loss')

        # Critic (value) Loss
        with tf.name_scope('critic_train'):
            td_error = self.td_target_ - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
            #critic_loss = tf.reduce_mean(tf.square(td_error * self.critic_correction_), name='critic_loss')

        return actor_loss, critic_loss, entropy

    def build_gradient(self):
        # Gradient
        with tf.name_scope('local_grad'):
            a_grads = tf.gradients(self.actor_loss, self.a_vars)
            c_grads = tf.gradients(self.critic_loss, self.c_vars)
            if self.grad_clip_norm:
                a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                           for grad, var in self.a_grads if grad is not None]
                c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                           for grad, var in self.c_grads if grad is not None]
        return a_grads, c_grads

    def build_pipeline(self):
        # Sync with Global Network
        actor_optimizer = self.global_network.actor_optimizer
        critic_optimizer = self.global_network.critic_optimizer
        with tf.name_scope('sync'):
            # Pull global weights to local weights
            with tf.name_scope('pull'):
                pull_a_vars = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars, self.global_network.a_vars)]
                pull_c_vars = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, self.global_network.c_vars)]

            # Push local weights to global weights
            with tf.name_scope('push'):
                update_a = actor_optimizer.apply_gradients(zip(self.a_grads, self.global_network.a_vars))
                update_c = critic_optimizer.apply_gradients(zip(self.c_grads, self.global_network.c_vars))
        return pull_a_vars, pull_c_vars, update_a, update_c

    # Update global network with local gradients
    def update_global(self, state, gps_state, goal,
                      action, adv, td_target, is_weight):

        # update Sequence
        feed_dict = {self.local_state_: np.stack(state),
                     self.gps_state_: np.stack(gps_state),
                     self.goal_state_: np.stack(goal),
                     self.action_: action,
                     self.adv_: adv,
                     self.td_target_: td_target,
                     self.actor_correction_: is_weight}

        ops = [self.actor_loss, self.critic_loss, self.entropy, self.update_a, self.update_c]
        actor_loss, critic_loss, entropy, _, __ = self.sess.run(ops, feed_dict)
        return actor_loss, critic_loss, entropy

    def pull_global(self):
        self.sess.run([self.pull_a_vars, self.pull_c_vars])

    # Forward Propagation
    def get_action(self, state, gps, goal):
        feed_dict = {self.local_state_: np.stack(state),
                     self.gps_state_: np.stack(gps),
                     self.goal_state_: np.stack(goal)}

        a_probs = self.sess.run(self.actor, feed_dict)
        action = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs]

        return action, a_probs

    def get_critic(self, state, gps, goal):
        feed_dict = {self.local_state_: np.stack(state),
                     self.gps_state_: np.stack(gps),
                     self.goal_state_: np.stack(goal)}

        critic = self.sess.run(self.critic, feed_dict)

        return critic.tolist()
