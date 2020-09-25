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
    def __init__(self, input_shape, action_size=5, atoms=128,
            trainable=True):
        super(V4COMA_d, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size, activation='relu')
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Loss
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        #self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
        #        reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Actor
        net = self.feature_layer(inputs)
        net = self.actor_dense1(net)
        net = self.actor_dense2(net)
        softmax_logits = self.softmax(net)
        log_logits = self.log_softmax(net)
        action = tf.squeeze(tf.random.categorical(log_logits, 1, dtype=tf.int32))

        actor = {'softmax': softmax_logits, 'log_softmax': log_logits, 'action': action}

        return actor


class V4COMA_c(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, num_agent, action_size=5, atoms=128,
                 trainable=True):
        super(V4COMA_c, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape)
        self.action_space = action_size**num_agent

        # Critic
        self.critic_dense1 = layers.Dense(units=atoms, activation='relu')
        self.critic_dense2 = layers.Dense(units=self.action_space)

        # Loss Operations
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        #self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
        #        reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.print_summary()

    def call(self, inputs, actions):
        # dim[inputs] : [num_batch, num_agent, lx, ly, ch]
        states = tf.unstack(inputs, axis=1)

        arr = []
        for idx in range(self.num_agent):
            net = self.feature_layer(states[idx])
            arr.append(net)

        net = tf.concat(arr, axis=1)

        # Critic  
        net = self.critic_dense1(net)
        critic = self.critic_dense2(net)

        action_one_hot = tf.one_hot(actions, self.action_space)
        critic_a = tf.reduce_sum(critic * action_one_hot, axis=1)

        return critic, critic_a

@tf.function
def loss_central(model, metastate, metaaction, td_target):
    Q, Q_s_a = model(metastate, metaaction)

    # Critic - TD Difference
    critic_mse = model.mse_loss_mean(Q_s_a, td_target)

    info = {'critic_loss': critic_mse}

    return critic_mse, info

@tf.function
def loss_decentral(model, state, action, advantage):
    # Run Model
    actor = model(state)
    pi = actor['softmax']

    # Actor Loss
    action_one_hot = tf.one_hot(action, model.action_size)
    pi_a = tf.reduce_sum(pi * action_one_hot, axis=1)
    actor_loss = -tf.reduce_sum(advantage * tf_log(pi_a))

    info = {'actor_loss': actor_loss}

    return actor_loss, info

def train(model, loss, optimizer, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val, info

