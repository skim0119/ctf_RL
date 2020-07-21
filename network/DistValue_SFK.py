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
from network.model_V4 import V4Decentral, V4INVDecentral
from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

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
    # Low gain weight more on estimated value
    # High gain weight more on measured(observed) distribution
    var1 = tf.math.exp(log_var1)
    var2 = tf.math.exp(log_var2)
    if gain is None:
        K = var1 / (var1 + var2) # Gain
    else:
        K = gain
    mu = mu1 + K * (mu2 - mu1)
    var = (1 - K) * var1
    return mu, tf.math.log(var)


class V4SFK_DECENTRAL(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5,
            trainable=True, name='DecPPO'):
        super(V4SFK_DECENTRAL, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4Decentral(input_shape, action_size)

        # Actor
        self.actor_dense1 = layers.Dense(128)
        self.actor_dense2 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Phi
        self.phi_dense1 = layers.Dense(units=64, activation='elu', name='phi')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01))

        # Psi
        self.psi_dense1 = layers.Dense(units=64, activation='elu')
        self.psi_dense2 = layers.Dense(units=64, activation='linear', name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Run full network
        obs = inputs[0]
        h = inputs[1]

        # Feature Encoding
        net = self.feature_layer(obs)
        net = tf.concat([net, h], axis=1)

        # Actor
        actor_net = self.actor_dense1(net)
        actor_net = self.actor_dense2(actor_net)
        softmax_logits = self.softmax(actor_net)
        log_logits = self.log_softmax(actor_net)

        # SF
        phi = self.phi_dense1(net)
        r_pred = self.successor_weight(phi)

        psi = self.psi_dense1(phi)
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi, training=False)

        SF = {'reward_predict': r_pred,
              'phi': phi,
              'psi': psi,
              'critic': critic}
        actor = {'softmax': softmax_logits,
                 'log_softmax': log_logits}

        return actor, SF


class V4SFK_CENTRAL(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, atoms,
                 trainable=True, name='SFK'):
        super(V4SFK_CENTRAL, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape)
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

        # Layers
        self.flat = layers.Flatten()

    def print_summary(self):
        self.feature_layer.print_summary()
        #self.summary()

    def inference_network(self, inputs):
        state = inputs[0]
        #b_mean = inputs[1]
        #b_log_var = inputs[2]

        # Encoder
        net = self.feature_layer(state)
        q_mu = self.z_mean(net)
        q_log_var = self.z_log_var(net)
        q_var = tf.math.exp(q_log_var)
        q_sigma = tf.math.sqrt(q_var)
        q_z = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)
        
        return q_z, q_mu, q_log_var

    def generative_network(self, q_z):
        z = q_z.sample()
        p_x_given_z_logits = self.decoder(z)
        # p_x_given_z = tfp.distributions.Bernoulli(logits=p_x_given_z_logits) # Bernoulli for binary image
        p_x_given_z = tfp.distributions.MultivariateNormalDiag(
                #tf.reshape(p_x_given_z_logits,[-1]), scale_identity_multiplier=0.05)
                self.flat(p_x_given_z_logits), scale_identity_multiplier=0.05)
        return p_x_given_z, p_x_given_z_logits

    def call(self, inputs):
        state = inputs[0]
        b_mean = inputs[1]
        b_log_var = inputs[2]

        # Encoder
        n_sample = state.shape[0]
        q_z, z_mean, z_log_var = self.inference_network(inputs)

        # Decoder
        p_x_given_z, z_decoded = self.generative_network(q_z)

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

        feature = {'latent': z,
                   'z_mean': z_mean,
                   'z_log_var': z_log_var,
                   'z_decoded': z_decoded,
                   'q_z': q_z,
                   'p_x_given_z': p_x_given_z}
        SF = {'reward_predict': r_pred,
              'phi': phi,
              'psi': psi,
              'critic': critic}
        pred_feature = {'pred_z_mean': pred_mean,
                        'pred_z_log_var': pred_log_var}

        return SF, feature, pred_feature
    
@tf.function
def loss_reward_central(model, state, reward, b_mean, b_log_var, training=True):
    # Critic - Reward Prediction
    inputs = [state, b_mean, b_log_var]
    SF, feature, pred_feature = model(inputs, training=training)
    r_pred = SF['reward_predict']
    reward = tf.cast(reward, tf.float32)
    reward_mse = tf.reduce_mean(tf.square(reward - r_pred))
    return reward_mse, {'reward_mse': reward_mse}

@tf.function
def loss_central_critic(model, state, b_mean, b_log_var, td_target, next_mean, next_log_var,
        psi_beta=1.0, kalman_pred_beta=0.5, beta_kl=1e-4, elbo_beta=0.3,
        gamma=0.98, training=True):
    inputs = [state, b_mean, b_log_var]
    SF, feature, pred_feature = model(inputs, training=training)

    # Critic - TD Difference
    psi = SF['psi']
    td_target = tf.cast(td_target, tf.float32)
    td_error = td_target - psi
    psi_mse = tf.reduce_mean(tf.square(td_error))

    # Kalman - predictor training
    pred_mean = pred_feature['pred_z_mean']
    pred_log_var = pred_feature['pred_z_log_var']
    pred_loss = tf.reduce_mean(tf.square(pred_mean - next_mean)) + tf.reduce_mean(tf.square(pred_log_var - next_log_var))

    # VAE ELBO loss
    p_z = tfp.distributions.Normal(loc=np.zeros(model.atoms, dtype=np.float32),
                                   scale=np.ones(model.atoms, dtype=np.float32))
    q_z = feature['q_z']
    p_x_given_z = feature['p_x_given_z']
    e_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(model.flat(state))) # Reconstruction term
    kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    elbo = tf.reduce_mean(e_log_likelihood - kl_loss, axis=0)

    total_loss = psi_beta*psi_mse + kalman_pred_beta*pred_loss - elbo_beta*elbo

    info = {
        'psi_mse': psi_mse,
        'pred_loss': pred_loss,
        'elbo': elbo,
        }

    return total_loss, info

#@tf.function
def loss_ppo(model, state, belief, old_log_logit, action, td_target, advantage,
         eps=0.2, entropy_beta=0.05, psi_beta=0.5,
         gamma=0.98, training=True):
    num_sample = state.shape[0]

    # Run Model
    pi, v = model([state, belief])
    actor = pi['softmax']
    psi = v['psi']
    log_logits = pi['log_softmax']

    # Entropy
    entropy = -tf.reduce_mean(actor * tf_log(actor), name='entropy')

    # Psi Loss
    td_target = tf.cast(td_target, tf.float32)
    td_error = td_target - psi
    psi_mse = tf.reduce_mean(tf.square(td_error), name='critic_loss')

    # Actor Loss
    action_OH = tf.one_hot(action, model.action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_OH, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

    # Clipped surrogate function
    ratio = tf.exp(log_prob - old_log_prob) # precision
    #ratio = log_prob / old_log_prob
    advantage = tf.cast(advantage, tf.float32)
    surrogate = ratio * advantage
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    total_loss = actor_loss + psi_beta*psi_mse - entropy_beta*entropy
    info = {'actor_loss': actor_loss,
            'psi_loss': psi_mse,
            'entropy': entropy}

    return total_loss, info

#@tf.function
def loss_multiagent_critic(model, states_list, belief, value_central, mask, training=True):
    # states_list and belief should be given in flat format
    num_env = mask.shape[0]
    num_agent = mask.shape[1]
    pi, v = model([states_list, belief], training=training)
    critics = tf.reshape(v['critic'], mask.shape) * tf.cast(mask, tf.float32)
    group_critic = tf.reduce_sum(critics, axis=1)
    mse = tf.reduce_mean(tf.square(value_central - group_critic))
    return mse, {'ma_critic': mse}

def train(model, loss, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])

    return loss_val, info

