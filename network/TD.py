import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.attention import Non_local_nn
from network.model_V4 import V4, V4INV
from utility.utils import store_args

import numpy as np


class V4TD(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size,
                 trainable=True, name='TemporalValue'):
        super(V4TD, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.dense1 = layers.Dense(units=64, activation='elu')
        self.dense2 = layers.Dense(units=1, name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def print_summary(self):
        self.feature_layer.print_summary()

    def call(self, inputs):
        state = inputs[0]

        # Encoder
        n_sample = state.shape[0]
        net = self.feature_layer(state)
        net = self.dense1(net)
        critic = self.dense2(net)

        return critic

def loss(model, state, reward, done, next_state,
         td_target,
         gamma=0.98, training=True):
    num_sample = state.shape[0]

    # Run Model
    inputs = [state]
    v_out = model(inputs, training=training)

    # Critic - TD Difference
    td_error = td_target - v_out
    critic_loss = tf.reduce_mean(tf.square(td_error))

    return critic_loss

def train(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val = loss(model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_val

