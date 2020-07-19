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

class PPO_SF(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, phi_n=16, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, name='PPO'):
        super(PPO_SF, self).__init__(name=name)

        # Feature Encoder
        self.latent_net = tf.keras.Sequential([
                layers.Conv2D(filters=16, kernel_size=5, strides=2, activation='relu'),
                layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
                layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
                layers.Flatten(),
                layers.Dense(units=128, activation='relu'),
            ])

        # Actor
        self.actor_dense1 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')

        # Successor Feature
        self.N = phi_n
        self.successor_layer = layers.Dense(1, activation='linear', name='successor', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.phi_net = tf.keras.Sequential([
                layers.Dense(self.N, activation='relu', name='phi'),
                layers.Dropout(0.1),
            ], name='reward_prediction')


        self.psi_net = tf.keras.Sequential([
                layers.Dense(self.N, activation='relu', name='phi'),
                layers.Dropout(0.1),
            ], name='value')

    def call(self, inputs):
        # state_input : [None(batch_size), 39, 39, 6*keep_frame]

        state_input = inputs

        # Encoding
        net = self.latent_net(state_input)

        # Actor
        self.logits = self.actor_dense1(net) 
        self.actor = self.softmax(self.logits)
        self.log_logits = tf.nn.log_softmax(self.logits)

        # SF
        self.phi = self.phi_net(net)
        sf_reward = self.successor_layer(self.phi)
        self.sf_reward = tf.reshape(sf_reward, [-1])

        # Value
        self.psi = self.psi_net(net)
        critic = self.successor_layer(self.psi)
        self.critic = tf.reshape(critic, [-1])

        return self.actor, self.logits, self.log_logits, self.critic, self.phi, self.sf_reward, self.psi

    @property
    def get_feature_variables(self):
        return self.latent_net.variables

    @property
    def get_actor_variables(self):
        return self.get_feature_variables+self.actor_dense1.variables

    @property
    def get_phi_variables(self):
        return self.get_feature_variables+self.phi_net.variables+self.successor_layer.variables

    @property
    def get_psi_variables(self):
        return self.get_feature_variables+self.psi_net.variables

    def build_loss(self, old_log_logit, action, advantage, td_target, result):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

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

            if self.entropy_beta != 0:
                actor_loss = actor_loss + entropy * self.entropy_beta

            # SF Loss
            with tf.name_scope('sf_loss'):
                td_error = td_target - self.psi
                sf_diff = []
                for i in range(self.N):
                    oh = tf.reshape(tf.one_hot(i, self.N), [self.N, 1])
                    mse = tf.reduce_mean(tf.square(tf.matmul(td_error,oh)))
                    sf_diff.append(mse)
                sf_loss = tf.reduce_sum(sf_diff, name='sf_loss')

            # Reward Supervised Training
            reward_loss = tf.keras.losses.MSE(result, self.sf_reward)

            self.actor_loss = actor_loss
            self.critic_loss = sf_loss
            self.entropy = entropy
            self.reward_loss = reward_loss

        return actor_loss, sf_loss, reward_loss


class PPO_SF_softmax(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, name='PPO'):
        super(PPO_SF, self).__init__(name=name)

        # Feature Encoder
        #self.conv1 = layers.SeparableConv2D(
        #        filters=16,
        #        kernel_size=5,
        #        strides=3,
        #        padding='valid',
        #        depth_multiplier=2,
        #        activation='relu',
        #    )
        self.conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=2, activation='relu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(units=128, activation='relu')

        # Actor
        self.actor_dense1 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')

        # Successor Feature
        self.N = 16
        self.phi_dense1 = layers.Dense(self.N, activation='relu', name='phi')
        self.successor_layer = layers.Dense(3, activation='linear', name='reward_prediction', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.spectrum_adder = layers.Dense(1, trainable=False, name='spectrum', use_bias=False, activation='linear',
                kernel_initializer=tf.constant_initializer([-1,0,1]))
        self.reward_layer = tf.keras.layers.Multiply()

        #self.psi_dense1 = layers.Dense(128, activation='relu')
        self.psi_dense2 = layers.Dense(self.N, activation='relu', name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.0001))

    def call(self, inputs):
        # state_input : [None(batch_size), 39, 39, 6*keep_frame]
        # done_state : [None(batch_size), 1]

        state_input, done_state = inputs

        # Encoding
        net = state_input
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.flat(net)
        net = self.dense1(net)

        # Actor
        logits = self.actor_dense1(net) 
        actor = self.softmax(logits)
        log_logits = tf.nn.log_softmax(logits)

        # SF
        phi = self.phi_dense1(net)
        sf_result = self.successor_layer(phi)
        sf = self.spectrum_adder(sf_result)
        sf_reward = tf.reshape(self.reward_layer([sf, done_state]), [-1])

        # Value
        #psi = self.psi_dense1(net)
        #psi = self.psi_dense2(psi)
        psi = self.psi_dense2(net)
        critic = self.successor_layer(psi)
        critic = self.spectrum_adder(critic)
        critic = tf.reshape(critic, [-1])

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.phi = phi
        self.sf_reward = sf_reward
        self.sf_result = sf_result
        self.psi = psi

        return actor, logits, log_logits, critic, phi, sf_reward, psi

    @property
    def get_feature_variables(self):
        return self.conv1.variables+self.conv2.variables+self.dense1.variables

    @property
    def get_actor_variables(self):
        return self.get_feature_variables+self.actor_dense1.variables

    @property
    def get_phi_variables(self):
        return self.get_feature_variables+self.phi_dense1.variables+self.successor_layer.variables

    @property
    def get_psi_variables(self):
        return self.get_feature_variables+self.psi_dense2.variables#self.psi_dense1.variables+self.psi_dense2.variables

    def build_loss(self, old_log_logit, action, advantage, td_target, result):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

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

            if self.entropy_beta != 0:
                actor_loss = actor_loss + entropy * self.entropy_beta

            # SF Loss
            with tf.name_scope('sf_loss'):
                td_error = td_target - self.psi
                sf_diff = []
                for i in range(self.N):
                    oh = tf.reshape(tf.one_hot(i, self.N), [self.N, 1])
                    mse = tf.reduce_mean(tf.square(tf.matmul(td_error,oh)))
                    sf_diff.append(mse)
                sf_loss = tf.reduce_sum(sf_diff, name='sf_loss')
            #critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Reward Supervised Training
            result = tf.one_hot(result, 3, dtype=tf.float32)
            reward_loss = tf.keras.losses.categorical_crossentropy(result, self.sf_result)
            #reward_loss = tf.keras.losses.MSE(reward, self.sf_reward)

            self.actor_loss = actor_loss
            self.critic_loss = sf_loss
            self.entropy = entropy
            self.reward_loss = reward_loss

        return actor_loss, sf_loss, reward_loss

if __name__=='__main__':
    network = PPO_SF()
    #z = network(tf.placeholder(tf.float32, [None, 4, 39, 39, 6]))
    first_batch = tf.zeros((1,39,39,12))
    z = network(first_batch)
    print(z)
    network.summary()
    print(network.layers)
    for lr in network.layers:
        print(lr.name, lr.input_shape, lr.output_shape)
    print(network.trainable_variables)
