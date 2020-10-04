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

class VDNNet(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5,
            trainable=True, name='VDN'):
        super(V4PPO, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)

        # Advantage
        self.advantage_dense1 = layers.Dense(256, activation='relu')
        self.advantage_dense2 = layers.Dense(action_size, activation='linear')

        # Critic
        self.critic_dense1 = layers.Dense(256, activation='relu')
        self.critic_dense2 = layers.Dense(1, activation='linear', use_bias=False)

    def call(self, inputs):
        shared_net = self.feature_layer(inputs)

        # Advantage
        net = self.advantage_dense1(shared)
        adv = self.advantage_dense2(net)

        # Critic
        net = self.critic_dense1(shared)
        critic = self.critic_dense2(net)

        qvals = critic + (adv - tf.reduce_mean(adv, axis=1))

        action = tf.math.argmax(qvals, axis=1)
        critic = tf.math.max(qvals, axis=1)

        return action, qvals, critic

def loss_critic(models, state, td_target, agent_type_index, gamma=0.98):
    # Run Model
    central_critic = None
    for idx, atype in enumerate(agent_type_index):
        model = models[atype]
        _, _, critic = model(state[:,idx,...])

        if central_critic is None:
            central_critic = critic
        else:
            central_critic += critic

    # Critic Loss
    loss = tf.reduce_mean(tf.square(central_critic - td_target))
    info = {'critic_loss': loss}

    return loss, info

def train_critic(models, optimizer, inputs):
    with tf.GradientTape() as tape:
        total_loss, info = loss_critic(models, **inputs)
    variables = []
    for t in model:
        variables += t.trainable_variables
    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, variables)
        if grad is not None])

    return total_loss, info

