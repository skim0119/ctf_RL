import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.model_V4_30 import V4
from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

import numpy as np

class V4PG(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5,
            trainable=True, name='PG'):
        super(V4PG, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.pi_layer = V4(input_shape, action_size)

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Actor
        net = self.pi_layer(inputs)
        net = self.actor_dense1(net)
        net = self.actor_dense2(net)
        actor = self.softmax(net)
        log_logits = self.log_softmax(net)

        return actor, log_logits

def loss(model, state, old_log_logit, action, advantage,
         eps=0.2, entropy_beta=0.05, 
         training=True, return_losses=False):
    num_sample = state.shape[0]

    # Run Model
    actor, log_logits = model(state)

    # Entropy
    H = -tf.reduce_mean(actor * tf_log(actor), axis=-1)
    mean_entropy = tf.reduce_mean(H)

    # Actor Loss
    action_OH = tf.one_hot(action, model.action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_OH, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

    # Clipped surrogate function
    ratio = tf.exp(log_prob - old_log_prob) # precision
    #ratio = log_prob / old_log_prob
    surrogate = ratio * advantage
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    total_loss = actor_loss - entropy_beta*mean_entropy

    info = None
    if return_losses:
        info = {'actor_loss': actor_loss,
                'entropy': mean_entropy}

    return total_loss, info

def get_gradient(model, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        total_loss, info = loss(model, **inputs, **hyperparameters, training=True, return_losses=True)
    grads = tape.gradient(total_loss, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return grads, info

def train(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        total_loss, info = loss(model, **inputs, **hyperparameters, training=True, return_losses=True)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])

    return total_loss, info
