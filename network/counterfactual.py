import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.model_V4_30 import V4, V4INV
from network.model_V4_30 import V4Decentral, V4INVDecentral

from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

import numpy as np


class V4COMA_d(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5, atoms=8,
            trainable=True):
        super(V4COMA_d, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.pi_layer = V4(input_shape, action_size)

        # Critic
        self.critic_dense1 = layers.Dense(units=atoms, activation='relu')
        self.sf_q_weight = layers.Dense(units=action_size, activation='linear', use_bias=False,)

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size, activation='relu')
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Loss
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Run full network
        obs = inputs[0]

        # Feature Encoding SF-phi
        net = self.feature_layer(obs)

        # Actor
        net = self.pi_layer(obs)
        #net = tf.concat([net, tf.stop_gradient(phi)], axis=1)
        net = self.actor_dense1(net)
        net = self.actor_dense2(net)
        softmax_logits = self.softmax(net)
        log_logits = self.log_softmax(net)
        action = tf.squeeze(tf.random.categorical(log_logits, 1, dtype=tf.int32))

        # Critic
        net = self.critic_dense1(net)
        q = self.sf_q_weight(net)
        critic = tf.gather(q, action, axis=1)
        iq = tf.reduce_sum(q * tf.stop_gradient(softmax_logits), axis=1) - critic

        actor = {'softmax': softmax_logits, 'log_softmax': log_logits, 'action': action}
        SF = {'critic': critic, 'Q': q, 'revQ': iq}

        return actor, SF


class V4COMA_c(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, atoms,
                 trainable=True):
        super(V4COMA_c, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape)

        # Critic
        self.critic_dense1 = layers.Dense(units=atoms, activation='relu')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False)

        # Loss Operations
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.print_summary()

    def call(self, inputs):
        state = inputs

        # Encoder
        n_sample = state.shape[0]
        net = self.feature_layer(state)

        # Critic  
        net = self.critic_dense1(net)
        critic = self.successor_weight(net)

        feature = {'latent': net}
        SF = {'critic': critic}

        return SF, feature 

@tf.function
def loss_central(model, state, td_target_c,
        critic_beta):
    inputs = state
    SF, feature = model(inputs)

    # Critic - TD Difference
    critic_mse = model.mse_loss_mean(td_target_c, SF['critic'])

    total_loss = critic_beta * critic_mse
    info = {'critic_mse': critic_mse}

    return total_loss, info

@tf.function
def loss_ppo(model, state, old_log_logit, action, old_value, advantage, td_target_c, next_state,
        eps, entropy_beta, q_beta):
    num_sample = state.shape[0]

    # Run Model
    pi, v = model([state, action])
    pi_next, v_next = model([next_state, action])
    actor = pi['softmax']
    log_logits = pi['log_softmax']

    # Entropy
    H = -tf.reduce_sum(actor * tf_log(actor), axis=-1) # Entropy H of each sample
    mean_entropy = tf.reduce_mean(H)

    # Actor Loss
    action_one_hot = tf.one_hot(action, model.action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_one_hot, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_one_hot, 1)
    ratio = tf.exp(log_prob - old_log_prob) # precision: log_prob / old_log_prob
    surrogate = ratio * advantage # Clipped surrogate function
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    # Q - Loss
    q = v['Q']
    q_a = tf.reduce_sum(q * action_one_hot, 1)
    q_loss = tf.reduce_mean(tf.square(q_a - td_target_c))

    total_loss = actor_loss
    total_loss -= entropy_beta*mean_entropy
    total_loss += q_beta*q_loss

    # Log
    info = {'actor_loss': actor_loss,
            'entropy': mean_entropy,
            'q_loss': q_loss,
            }

    return total_loss, info

def get_gradient(model, loss, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return grads, info

def train(model, loss, optimizer, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val, info

