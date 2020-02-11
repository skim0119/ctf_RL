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
import numpy as np


class PPO_SF(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, name='PPO',N=5, keep_frames=1):
        super(PPO_SF, self).__init__(name=name)

        # Feature Encoder
        self.conv1 = layers.SeparableConv2D(
                filters=16,
                kernel_size=5,
                strides=3,
                padding='valid',
                depth_multiplier=2,
                activation='relu',
            )
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(units=256, activation='relu')

        # Actor
        self.actor_dense1 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')

        # Successor Feature
        self.N = N
        self.phi_dense1 = layers.Dense(256, activation='relu')
        self.phi_dense2 = layers.Dense(self.N, activation='relu', name='phi')
        self.successor_layer = layers.Dense(1, activation='linear', name='reward_prediction', use_bias=False)
        self.psi_dense1 = layers.Dense(256, activation='relu')
        self.psi_dense2 = layers.Dense(self.N, activation='relu', name='psi')

        #State Prediction
        self.sp_dense = layers.Dense(units=4*4*16, name="State_Prediction_Dense")
        self.reshape = layers.Reshape(target_shape=(4,4,16))
        self.sp_conv1 = layers.Conv2DTranspose(filters=32,kernel_size=3, strides=3,activation='relu', name="State_Prediction_Conv1")
        self.sp_conv2 = layers.Conv2DTranspose(filters=16,kernel_size=5, strides=3,padding='valid',activation='relu', name="State_Prediction_Conv2")
        self.sp_conv3 = layers.Conv2DTranspose(filters=6*keep_frames,kernel_size=2, strides=1,padding='valid',activation='relu', name="State_Prediction_Conv3")


    def call(self, inputs):
        # state_input : [None, 39, 39, 6*keep_frame]

        net = inputs
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.flat(net)
        net = self.dense1(net)
        phi = self.phi_dense1(net)
        phi = self.phi_dense2(phi)

        logits = self.actor_dense1(phi)
        actor = self.softmax(logits)
        log_logits = tf.nn.log_softmax(logits)

        sf_reward = self.successor_layer(phi)
        sf_reward = tf.reshape(sf_reward, [-1])

        psi = self.psi_dense1(phi)
        psi = self.psi_dense2(psi)
        critic = self.successor_layer(psi)
        critic = tf.reshape(critic, [-1])

        sp = self.sp_dense(phi)
        sp = self.reshape(sp)
        sp = self.sp_conv1(sp)
        sp = self.sp_conv2(sp)
        sp = self.sp_conv3(sp)
        # state_prediction = sp - (tf.sin(2*np.pi*sp)-tf.sin(4*np.pi*sp)/2)/np.pi # Added to round values. Sawtooth Wave
        state_prediction = sp - (tf.sin(2*np.pi*sp))/(2*np.pi) # Added to round values. Sawtooth Wave
        # state_prediction = tf.round(sp)

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.phi = phi
        self.sf_reward = sf_reward
        self.psi = psi

        self.state_prediction = state_prediction

        return actor, logits, log_logits, critic, phi, sf_reward, psi, state_prediction

    @property
    def get_feature_variables(self):
        return self.conv1.variables+self.conv2.variables+self.dense1.variables+self.phi_dense1.variables+self.phi_dense2.variables

    @property
    def get_actor_variables(self):
        return self.get_feature_variables+self.actor_dense1.variables

    @property
    def get_phi_variables(self):
        return self.get_feature_variables + self.successor_layer.variables

    @property
    def get_psi_variables(self):
        return self.get_feature_variables+self.psi_dense1.variables+self.psi_dense2.variables

    @property
    def get_state_pred_variables(self):
        return self.get_feature_variables+self.sp_dense.variables+self.sp_conv1.variables+self.sp_conv2.variables+self.sp_conv3.variables

    def build_loss(self, old_log_logit, action, advantage, td_target, reward, state_next,phi):
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
                td_error = phi + td_target - self.psi
                sf_diff = []
                for i in range(self.N):
                    oh = tf.reshape(tf.one_hot(i, self.N), [self.N, 1])
                    mse = tf.reduce_mean(tf.square(tf.matmul(td_error,oh)))
                    sf_diff.append(mse)
                sf_loss = tf.reduce_sum(sf_diff, name='sf_loss')
            #critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Reward Supervised Training
            reward_loss = tf.keras.losses.MSE(reward, self.sf_reward)

            #State prediction loss
            state_loss = tf.reduce_mean(tf.math.pow(tf.subtract(state_next, self.state_prediction),3))
            print(state_loss)

            self.actor_loss = actor_loss
            self.critic_loss = sf_loss
            self.entropy = entropy
            self.reward_loss = reward_loss
            self.state_loss = state_loss

        return actor_loss, sf_loss, reward_loss, state_loss

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
