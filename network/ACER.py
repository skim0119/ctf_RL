import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import random

from utility.utils import retrace, store_args


class ACER:
    @store_args
    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4, grad_clip_norm=0,
                 entropy_beta=0.001, sess=None, globalAC=None):
        """ Initialize AC network and required parameters """
        
        with tf.variable_scope(scope):
            # Optimizer
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor)
            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic)

            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input_ = tf.placeholder(shape=in_size, dtype=tf.float32, name='state')

            # get the parameters of actor and critic networks
            self.actor, self.critic, self.a_vars, self.c_vars = self.build_network(state_input_)

            # Local Network
            if scope != 'global':
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                #self.reward_ = tf.placeholder(shape=[None],dtype=tf.float32, name='reward_holder')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                self.adv_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')
                self.retrace_ = tf.placeholder(shape=[None], dtype=tf.float32, name='retrace_holder')
                self.retrace_prod_ = tf.placeholder(shape=[None], dtype=tf.float32, name='retrace_prod_holder')

                # Actor Loss
                with tf.name_scope('actor_train'):
                    self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
                    action_OH = tf.one_hot(self.action_, self.action_size)
                    pi_t = tf.reduce_sum(self.actor * action_OH, 1)  # policy for corresponding state and action
                    exp_v = tf.log(pi_t) * self.adv_ * self.retrace_ + self.entropy_beta * self.entropy
                    self.actor_loss = -tf.reduce_mean(exp_v, name='actor_loss')  # or reduce_sum

                # Critic (value) Loss
                with tf.name_scope('critic_train'):
                    self.td_error = self.td_target_ - self.critic  # for gradient calculation (equal to advantages)
                    self.critic_loss = tf.reduce_mean(tf.square(self.td_error * self.retrace_prod_),
                                                      name='critic_loss')  # mse of td error

                # Gradient
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.actor_loss, self.a_vars)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_vars)
                    if self.grad_clip_norm:
                        self.a_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.a_grads if grad is not None]
                        self.c_grads = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.c_grads if grad is not None]

                # Sync with Global Network
                with tf.name_scope('sync'):
                    # Pull global weights to local weights
                    with tf.name_scope('pull'):
                        self.pull_a_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.a_vars, globalAC.a_vars)]
                        self.pull_c_vars_op = [local_var.assign(glob_var) for local_var, glob_var in zip(self.c_vars, globalAC.c_vars)]

                    # Push local weights to global weights
                    with tf.name_scope('push'):
                        self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_vars))
                        self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_vars))

    def build_network(self, input_):
        layer = layers.conv2d(input_, 32, [3, 3],
                              activation_fn=tf.nn.relu,
                              weights_initializer=layers.xavier_initializer_conv2d(),
                              biases_initializer=tf.zeros_initializer(),
                              padding='SAME')
        layer = layers.max_pool2d(layer, [2, 2])
        layer = layers.conv2d(layer, 64, [2, 2],
                              activation_fn=tf.nn.relu,
                              weights_initializer=layers.xavier_initializer_conv2d(),
                              biases_initializer=tf.zeros_initializer(),
                              padding='SAME')
        layer = layers.flatten(layer)

        with tf.variable_scope('actor'):
            actor = layers.fully_connected(layer, 64)
            actor = layers.fully_connected(self.actor, self.action_size,
                                           activation_fn=tf.nn.softmax)
            actor_argmax = tf.argmax(self.actor, axis=1, output_type=tf.int32, name='argmax')

        with tf.variable_scope('critic'):
            critic = layers.fully_connected(layer, 1,
                                            activation_fn=None)
            critic = tf.reshape(self.critic, (-1,))

        common_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/Conv')
        a_vars = common_vars + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')

        return actor, critic, a_vars, c_vars

    # Update global network with local gradients
    def update_global(self, state, action, adv, td_target, retrace, retrace_prod):
        # update Sequence
        feed_dict = {self.state_input_: np.stack(state),
                     self.action_: action,
                     self.adv_: adv,
                     self.td_target_: td_target,
                     self.retrace_: retrace,
                     self.retrace_prod_: retrace_prod}

        ops = [self.actor_loss, self.critic_loss, self.entropy, self.update_a_op, self.update_c_op]
        al, cl, etrpy, _, __ = self.sess.run(ops, feed_dict)
        return al, cl, etrpy

    def pull_global(self):
        self.sess.run([self.pull_a_vars_op, self.pull_c_vars_op])

     # Choose Action
    def get_ac(self, s):
        a_probs, critic = self.sess.run([self.actor, self.critic], {self.state_input_: s})
        action = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs]

        return action, a_probs, critic
