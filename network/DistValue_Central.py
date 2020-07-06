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

class V4DistVar(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size,
                 atoms,
                 trainable=True, name='DISTVAR'):
        super(V4DistVar, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)
        self.z_mean = layers.Dense(units=atoms)
        self.z_log_var = layers.Dense(units=atoms)

        # Decoding
        self.decoder = V4INV()

        # Phi
        #self.phi_dense1 = layers.Dense(units=atoms, activation='elu', name='phi')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01))

        # Psi
        self.psi_dense1 = layers.Dense(units=32, activation='elu')
        self.psi_dense2 = layers.Dense(units=atoms, activation='elu', name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def print_summary(self):
        self.feature_layer.print_summary()
        #self.summary()

    def call(self, inputs):
        # Encoder
        n_sample = inputs.shape[0]
        net = self.feature_layer(inputs)
        z_mean = self.z_mean(net)
        z_log_var = self.z_log_var(net)

        # e Sampling
        epsilon = tf.random.normal(
                shape=(n_sample, self.atoms),
                mean=0.,
                stddev=1.0,
                name='sampled_epsilon')
        z = z_mean + tf.math.exp(z_log_var)*epsilon

        # Decoder
        z_decoded = self.decoder(tf.stop_gradient(z))

        # Critic
        #net = tf.concat([z_mean, z_log_var], axis=-1)
        #phi = self.phi_dense1(net)
        phi = z
        r_pred = self.successor_weight(z)
        psi = self.psi_dense1(tf.stop_gradient(z))
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi)

        return critic, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi

def loss(model, state, reward, done, next_state,
         td_target,
         beta_kl=1e-4,
         gamma=0.98, training=True):
    num_sample = state.shape[0]

    # Run Model
    v_out, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi = model(state, training=training)

    # VAE ELBO Loss
    ce_loss = tf.keras.losses.binary_crossentropy(
            tf.reshape(state, [num_sample, -1]),
            tf.reshape(z_decoded, [num_sample, -1]))
    kl_loss = -beta_kl * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    elbo = tf.reduce_mean(ce_loss + kl_loss)

    # Critic - Reward Prediction
    reward_mse = tf.reduce_mean(tf.square(reward - r_pred))

    # Critic - TD Difference
    #td_error = td_target - v_out
    td_error = td_target - psi
    psi_mse = tf.reduce_mean(tf.square(td_error))

    critic_loss = reward_mse + psi_mse

    return critic_loss, elbo 

def train(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        sf_loss, elbo = loss(model, **inputs, **hyperparameters, training=True)
        loss_val = sf_loss + elbo
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_val, sf_loss, elbo

