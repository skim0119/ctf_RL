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

class V4PPO(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5,
            trainable=True, name='PPO'):
        super(V4PPO, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.pi_layer = V4(input_shape, action_size)

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Critic
        self.critic_dense1 = layers.Dense(64, activation='relu')
        self.critic_dense2 = layers.Dense(1, activation='linear')
        self.smoothed_pseudo_H = tf.Variable(1.0)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Actor
        net = self.pi_layer(inputs)
        actor_net = self.actor_dense1(net)
        actor_net = self.actor_dense2(actor_net)
        actor = self.softmax(actor_net)
        log_logits = self.log_softmax(actor_net)

        # Critic
        net = self.feature_layer(inputs)
        critic = self.critic_dense1(net)
        critic = self.critic_dense2(critic)
        critic = tf.reshape(critic, [-1])

        return actor, critic, log_logits

def loss(model, state, old_log_logit, action, advantage, td_target, old_value,
         eps=0.2, entropy_beta=0.05, critic_beta=0.5, 
         gamma=0.98, training=True, return_losses=False):
    num_sample = state.shape[0]

    # Run Model
    actor, critic, log_logits = model(state)

    # Entropy
    entropy = -tf.reduce_mean(actor * tf_log(actor), name='entropy')
    pseudo_H = tf.stop_gradient(
            tf.reduce_sum(actor*(1-actor), axis=-1))
    mean_pseudo_H = tf.reduce_mean(pseudo_H)
    smoothed_pseudo_H = model.smoothed_pseudo_H

    # Critic Loss
    #td_error = td_target - critic
    #critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')
    v_pred = critic
    v_pred_clipped = old_value + tf.clip_by_value(v_pred-old_value, -eps, eps)
    critic_mse = tf.maximum(
        tf.square(v_pred - td_target),
        tf.square(v_pred_clipped - td_target))
    critic_loss = tf.reduce_mean(critic_mse * tf.stop_gradient(smoothed_pseudo_H)) + tf.square(mean_pseudo_H-smoothed_pseudo_H)

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

    total_loss = actor_loss + critic_beta*critic_loss - entropy_beta*entropy

    info = None
    if return_losses:
        info = {'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'entropy': entropy}

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

