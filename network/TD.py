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
    def __init__(self, input_shape, action_size, atoms,
                 trainable=True, name='TemporalValue'):
        super(V4TD, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.z_mean = layers.Dense(units=atoms)
        self.z_log_var = layers.Dense(units=atoms, activation=tf.nn.softplus)

        # Decoding
        self.decoder = V4INV()

        self.dense1 = layers.Dense(units=32, activation='elu')
        self.dense2 = layers.Dense(units=1, name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def print_summary(self):
        self.feature_layer.print_summary()

    def inference_network(self, inputs):
        # Encoder
        n_sample = inputs.shape[0]
        net = self.feature_layer(inputs)
        q_mu = self.z_mean(net)
        q_log_var = self.z_log_var(net)
        q_var = tf.math.exp(q_log_var)
        q_sigma = tf.math.sqrt(q_var)
        q_z = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)
        
        return q_z

    def generative_network(self, q_z):
        z = q_z.sample()
        p_x_given_z_logits = self.decoder(z)
        p_x_given_z = tfp.distributions.Bernoulli(logits=p_x_given_z_logits)
        return p_x_given_z

    def call(self, inputs):
        state = inputs[0]

        # Encoder
        q_z = self.inference_network(state)
        z = q_z.sample()

        # Decoder
        p_x_given_z = self.generative_network(q_z)

        net = self.dense1(z)
        critic = self.dense2(net)

        return critic

def loss_decoder(model, state, beta_kl=1e-4, training=True):
    num_sample = state.shape[0]
    p_z = tfp.distributions.Normal(loc=np.zeros(model.atoms, dtype=np.float32),
                                   scale=np.ones(model.atoms, dtype=np.float32))
    q_z = model.inference_network(state)
    p_x_given_z = model.generative_network(q_z)
    e_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(state), [1,2,3])
    kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    elbo = tf.reduce_mean(e_log_likelihood - kl_loss, axis=0)
    return -elbo

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

def train(model, loss, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val = loss(model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])

    return loss_val

