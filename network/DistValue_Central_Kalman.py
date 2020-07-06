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


class KalmanPredictor(tf.keras.Model):
    @store_args
    def __init__(self, num_input, num_hidden):
        super(KalmanPredictor, self).__init__()
        self.nn = keras.Sequential([
            layers.Input(shape=[num_input*2]),
            layers.Dense(units=num_hidden, activation='relu'),
            layers.Dense(units=num_hidden, activation='relu'),
            layers.Dense(units=num_input*2)])

    def call(self, input):
        mu = input[0]
        log_var = input[1]
        # action = input[2]

        # NN
        h = tf.concat([mu, log_var], axis=-1)
        net = self.nn(h)
        mu_, log_var_ = tf.split(net, 2, axis=-1)

        # Prediction
        mu = mu + mu_
        log_var = tf.math.log(tf.math.exp(log_var) + tf.math.exp(log_var_))

        return mu, log_var

@tf.function
def kalman_corrector(mu1, log_var1, mu2, log_var2):
    # Assume mu1 and var1 are estimated distribution, 
    # mu2 and var2 are measured distribution
    var1 = tf.math.exp(log_var1)
    var2 = tf.math.exp(log_var2)
    K = var1 / (var1 + var2) # Gain
    mu = mu1 + K * (mu2 - mu1)
    var = (1 - K) * var1
    return mu, tf.math.log(var)


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

        # Kalman
        self.kalman_predictor = KalmanPredictor(atoms, 16)

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
        state = inputs[0]
        b_mean = inputs[1]
        b_log_var = inputs[2]

        # Encoder
        n_sample = state.shape[0]
        net = self.feature_layer(state)
        z_mean = self.z_mean(net)
        z_log_var = self.z_log_var(net)

        # Decoder
        epsilon1 = tf.random.normal(
                shape=(n_sample, self.atoms),
                mean=0.,
                stddev=1.0,
                name='sampled_epsilon')
        z1 = z_mean + tf.math.exp(z_log_var)*epsilon1
        z_decoded = self.decoder(tf.stop_gradient(z1))

        # Kalman Filter
        z_mean, z_log_var = kalman_corrector(z_mean, z_log_var, b_mean, b_log_var)
        epsilon = tf.random.normal(
                shape=(n_sample, self.atoms),
                mean=0.,
                stddev=1.0,
                name='sampled_epsilon')
        z = z_mean + tf.math.exp(z_log_var)*epsilon
        pred_mean, pred_log_var = self.kalman_predictor([z_mean, z_log_var])

        # Critic
        #net = tf.concat([z_mean, z_log_var], axis=-1)
        #phi = self.phi_dense1(net)
        phi = z
        r_pred = self.successor_weight(phi)
        psi = self.psi_dense1(tf.stop_gradient(z))
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi)

        return critic, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi, pred_mean, pred_log_var

def loss(model, state, reward, done, next_state,
         td_target, b_mean, b_log_var, next_mean, next_log_var,
         beta_kl=1e-4,
         gamma=0.98, training=True):
    num_sample = state.shape[0]

    # Run Model
    inputs = [state, b_mean, b_log_var]
    v_out, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi, pred_mean, pred_log_var = model(inputs, training=training)

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

    # Predictor Loss
    pred_loss = tf.reduce_mean(tf.square(pred_mean - next_mean)) + tf.reduce_mean(tf.square(pred_log_var - next_log_var))

    return critic_loss, elbo, pred_loss

def train(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        sf_loss, elbo, pred_loss = loss(model, **inputs, **hyperparameters, training=True)
        loss_val = sf_loss + elbo + pred_loss
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_val, sf_loss, elbo, pred_loss

