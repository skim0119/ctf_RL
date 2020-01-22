import os
import sys
sys.path.append('/home/namsong/github/raide_rl')

from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as layers

from network.attention import non_local_nn_2d
from network.attention import Non_local_nn

from utility.utils import store_args

from method.base import put_channels_on_grid


class PPO_LSTM_V1(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, critic_beta=0.5, name='PPO'):
        super(PPO_LSTM_V1, self).__init__(name=name)

        # Feature Encoder
        conv1 = layers.SeparableConv2D(
                filters=16,
                kernel_size=5,
                strides=3,
                padding='valid',
                depth_multiplier=2,
                activation='relu',
            )
        self.td_conv1 = layers.TimeDistributed(conv1, input_shape=(None,4,79,79,6))

        conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.td_conv2 = layers.TimeDistributed(conv2)

        flat = layers.Flatten()
        self.td_flat  = layers.TimeDistributed(flat)

        dense1 = layers.Dense(units=256, activation='relu')
        self.td_dense1 = layers.TimeDistributed(dense1)

        self.lstm1 = layers.LSTM(256, return_state=True)

        # Actor
        self.actor_dense1 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')

        # Critic
        self.critic_dense1 = layers.Dense(5, activation='relu')
        self.critic_dense2 = layers.Dense(1)

    def call(self, inputs):
        # state_input : [None, keepframe, 39, 39, 6]
        # prev_action : [None, 1]
        # prev_reward : [None, 1]
        # hidden : [[None, 256], [None, 256]]
        state_input, prev_action, prev_reward, hidden = inputs

        net = state_input
        net = self.td_conv1(net)
        net = self.td_conv2(net)
        net = self.td_flat(net)
        net = self.td_dense1(net)
        net = tf.concat([net, prev_action, prev_reward], axis=2)
        #net, state_h, state_c = self.lstm1(net, initial_state=hidden) # Continuation hidden state
        #net, state_h, state_c = self.lstm1(net) # Arbitrary (noisy) hidden state (not sure if implemented)
        net, state_h, state_c = self.lstm1(net, initial_state=None) # Zero hidden state
        hidden = [state_h, state_c]

        logits = self.actor_dense1(net) 
        actor = self.softmax(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic = self.critic_dense1(net)
        critic = self.critic_dense2(critic)
        critic = tf.reshape(critic, [-1])

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.hidden_state = hidden

        return actor, logits, log_logits, critic, hidden

    def reset_lstm(self):
        # Reset every lstm layer to initial state.
        self.lstm1.reset_states()

    def build_loss(self, old_log_logit, action, advantage, td_target):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

            # Critic Loss
            td_error = td_target - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

            # Clipped surrogate function (PPO)
            ratio = tf.exp(log_prob - old_log_prob)
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            total_loss = actor_loss
            if self.entropy_beta != 0:
                total_loss = actor_loss + entropy * self.entropy_beta
            if self.critic_beta != 0:
                total_loss = actor_loss + critic_loss * self.critic_beta

            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy

        return total_loss

if __name__=='__main__':
    network = PPO_LSTM_V1()
    #z = network(tf.placeholder(tf.float32, [None, 4, 39, 39, 6]))
    first_batch = tf.zeros((1,4,39,39,6))
    z = network(first_batch)
    print(z)
    network.summary()
    print(network.layers)
    for lr in network.layers:
        print(lr.name, lr.input_shape, lr.output_shape)
    print(network.trainable_variables)
