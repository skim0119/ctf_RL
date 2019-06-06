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


class A3C_LSTM(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.001,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(A3C_LSTM, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

        self._build_rnn_trainer()

    def reset_rnn(self, num_memory):
        self.rnn_state = [self.state_init for _ in range(num_memory)]
        self.batch_rnn_state = [s for s in self.rnn_state]

    def run_network(self, states):
        actions = []
        critics = []
        for idx, state in enumerate(states):
            feed_dict = {
                self.state_input: state[np.newaxis,:],
                self.state_in[0]: self.rnn_state[idx][0],
                self.state_in[1]: self.rnn_state[idx][1]
            }
            ops = [self.actor, self.critic, self.state_out]
            prob, critic, self.rnn_state[idx] = self.sess.run(ops, feed_dict)
            action = np.random.choice(self.action_size, p=prob[0] / sum(prob[0])) 
            actions.append(action)
            critics.append(critic[0])
        return actions, critics

    def update_global(self, state_input, action, td_target, advantage, idx):
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
                     self.advantage_: advantage,
                     self.state_in[0]: self.batch_rnn_state[idx][0],
                     self.state_in[1]: self.batch_rnn_state[idx][1]}
        self.sess.run(self.update_ops, feed_dict)

        ops = [self.actor_loss, self.critic_loss, self.entropy, self.lstm_loss, self.lstm_train]
        aloss, closs, entropy, lstm_loss, _ = self.sess.run(ops, feed_dict)

        return aloss, closs, entropy, lstm_loss


    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        with tf.variable_scope('actor'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[16, 32, 32],
                kernels=[5, 3, 2],
                pools=[2, 2, 2],
                flatten=True
            )
            encoded_net = layers.fully_connected(net, 256)

            #Recurrent network
            lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=256, state_is_tuple=True)
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            self.state_in = (c_in, h_in)
            step_size = tf.shape(self.state_input)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell,
                    tf.expand_dims(encoded_net,[0]), 
                    initial_state=state_in,
                    sequence_length=step_size,
                    time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            lstm_net = tf.reshape(lstm_outputs, [-1, 256])
            self.lstm_context = lstm_net

            net = tf.concat([encoded_net, lstm_net],1)

            net = layers.fully_connected(
                net, 128,
                activation_fn=tf.nn.elu,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

            actor = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=tf.nn.softmax)

        with tf.variable_scope('critic'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[16, 32, 32],
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

    def _build_rnn_trainer(self):
        trainer = tf.train.AdamOptimizer(1e-3)
        pred = self.lstm_context[:-1]
        #pred = tf.gather_nd(params=self.lstm_context, indices=[)
        label = self.encoded_state[1:]

        self.lstm_loss = tf.losses.mean_squared_error(label, pred)
        self.lstm_train = trainer.minimize(self.lstm_loss)
