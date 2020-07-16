import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

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
            #layers.Dense(units=num_hidden, activation='relu'),
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
def kalman_corrector(mu1, log_var1, mu2, log_var2, gain=None):
    # mu1 and var1 are estimated distribution, 
    # mu2 and var2 are measured distribution
    var1 = tf.math.exp(log_var1)
    var2 = tf.math.exp(log_var2)
    if gain is None:
        K = var1 / (var1 + var2) # Gain
    else:
        K = gain
    mu = mu1 + K * (mu2 - mu1)
    var = (1 - K) * var1
    return mu, tf.math.log(var)


class V4SFK(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size,
                 atoms,
                 trainable=True, name='SFK'):
        super(V4SFK, self).__init__(name=name)

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
        self.psi_dense2 = layers.Dense(units=atoms, activation='linear', name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def print_summary(self):
        self.feature_layer.print_summary()
        #self.summary()

    def inference_network(self, inputs):
        state = inputs[0]
        #b_mean = inputs[1]
        #b_log_var = inputs[2]

        # Encoder
        n_sample = state.shape[0]
        net = self.feature_layer(state)
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
        b_mean = inputs[1]
        b_log_var = inputs[2]

        # Encoder
        n_sample = state.shape[0]
        net = self.feature_layer(state)
        z_mean = self.z_mean(net)
        z_log_var = self.z_log_var(net)

        # Decoder
        eps_dec = tf.random.normal(
                shape=(n_sample, self.atoms),
                mean=0.,
                stddev=1.0,
                name='sampled_epsilon')
        z1 = z_mean + tf.math.exp(z_log_var)*eps_dec
        z_decoded = self.decoder(tf.stop_gradient(z1))

        # Kalman Filter
        z_mean, z_log_var = kalman_corrector(z_mean, z_log_var, b_mean, b_log_var, 0.1)
        eps = tf.random.normal(
                shape=(n_sample, self.atoms),
                mean=0.,
                stddev=1.0,
                name='sampled_epsilon')
        z = z_mean# + tf.math.exp(z_log_var)*eps
        pred_mean, pred_log_var = self.kalman_predictor([z_mean, z_log_var])

        # Critic
        #net = tf.concat([z_mean, z_log_var], axis=-1)
        #phi = self.phi_dense1(net)
        phi = z
        r_pred = self.successor_weight(phi)
        #psi = self.psi_dense1(tf.stop_gradient(z))
        psi = self.psi_dense1(phi)
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi, training=False)

        return critic, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi, pred_mean, pred_log_var
    
def loss_reward(model, state, reward, b_mean, b_log_var, training=True):
    # Critic - Reward Prediction
    inputs = [state, b_mean, b_log_var]
    _,_,_,_,_,_, r_pred, _,_,_  = model(inputs, training=training)
    reward_mse = tf.reduce_mean(tf.square(reward - r_pred))
    return reward_mse

def loss_predictor(model, state, b_mean, b_log_var, next_mean, next_log_var, training=True):
    # Predictor Loss
    inputs = [state, b_mean, b_log_var]
    _,_,_,_,_,_, _, _,pred_mean,pred_log_var  = model(inputs, training=training)
    pred_loss = tf.reduce_mean(tf.square(pred_mean - next_mean)) + tf.reduce_mean(tf.square(pred_log_var - next_log_var))
    return pred_loss

def loss_decoder(model, state, b_mean, b_log_var, beta_kl=1e-4, training=True):
    # VAE ELBO Loss
    num_sample = state.shape[0]
    inputs = [state, b_mean, b_log_var]
    #_,_,z_mean,z_log_var,z_decoded,_,_,_,_,_= model(inputs, training=training)
    p_z = tfp.distributions.Normal(loc=np.zeros(model.atoms, dtype=np.float32),
                                   scale=np.ones(model.atoms, dtype=np.float32))
    q_z = model.inference_network(inputs)
    p_x_given_z = model.generative_network(q_z)
    e_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(state), [1,2,3])
    kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    #e_log_likelihood = tf.keras.losses.binary_crossentropy(
    #        tf.reshape(state, [num_sample, -1]),
    #        tf.reshape(z_decoded, [num_sample, -1]))
    #kl_loss = beta_kl * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    elbo = tf.reduce_mean(e_log_likelihood - kl_loss, axis=0)
    return -elbo

def loss_psi(model, state, td_target, b_mean, b_log_var, gamma=0.98, training=True):
    # Run Model
    inputs = [state, b_mean, b_log_var]
    v_out, z, z_mean, z_log_var, z_decoded, phi, r_pred, psi, pred_mean, pred_log_var = model(inputs, training=training)

    # Critic - TD Difference
    #td_error = td_target - v_out
    td_error = td_target - psi
    psi_mse = tf.reduce_mean(tf.square(td_error))

    return psi_mse

def train(model, loss, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val = loss(model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])

    return loss_val

