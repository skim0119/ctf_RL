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

class V4SF_CVDC_DECENTRAL(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5, atoms=8,
            trainable=True, name='DecPPO'):
        super(V4SF_CVDC_DECENTRAL, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4Decentral(input_shape, action_size)

        # Decoder
        self.decoder = V4INVDecentral()

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='elu')
        self.actor_dense2 = layers.Dense(action_size, activation='elu')
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Phi
        self.phi_dense1 = layers.Dense(units=atoms, activation='sigmoid', name='phi')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False)

        # Psi
        self.psi_dense1 = layers.Dense(units=64, activation='elu')
        self.psi_dense2 = layers.Dense(units=atoms, activation='elu', name='psi',
                kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.psi_loss_func = tf.keras.losses.MeanSquaredError()
        self.critic_loss_func = tf.keras.losses.MeanSquaredError()

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Run full network
        obs = inputs[0]
        h = inputs[1]

        # Feature Encoding
        net = self.feature_layer(obs)
        net = tf.concat([net, h], axis=1)

        decoded_state = self.decoder(net)

        # Actor
        actor_net = self.actor_dense1(net)
        actor_net = self.actor_dense2(actor_net)
        softmax_logits = self.softmax(actor_net)
        log_logits = self.log_softmax(actor_net)

        # SF
        phi = self.phi_dense1(net)
        r_pred = self.successor_weight(phi)

        psi = self.psi_dense1(tf.stop_gradient(phi))
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi, training=False)

        SF = {'reward_predict': r_pred,
              'phi': phi,
              'psi': psi,
              'critic': critic,
              'decoded_state': decoded_state}
        actor = {'softmax': softmax_logits,
                 'log_softmax': log_logits}

        return actor, SF


class V4SF_CVDC_CENTRAL(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, atoms,
                 trainable=True, name='SF_CVDC'):
        super(V4SF_CVDC_CENTRAL, self).__init__(name=name)

        # Feature Encoding
        self.feature_layer = V4(input_shape)
        self.z_mean = layers.Dense(units=atoms, activation='softmax')
        self.z_log_var = layers.Dense(units=atoms, activation=tf.math.softplus)

        # Decoding
        self.decoder = V4INV()

        # Phi
        #self.phi_dense1 = layers.Dense(units=atoms, activation='elu', name='phi')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        # Psi
        self.psi_dense1 = layers.Dense(units=64, activation='elu')
        self.psi_dense2 = layers.Dense(units=atoms, activation='linear', name='psi')

        # UNREAL-RP
        output_bias = tf.keras.initializers.Constant([-2.302585, -0.223143, -2.302585])
        self.rp_dense = layers.Dense(units=3, activation='softmax',
                bias_initializer=output_bias)

        # Misc Layers
        self.flat = layers.Flatten()

        # Loss
        self.rp_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        self.reward_loss = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)
        self.critic_loss = tf.keras.losses.MeanSquaredError()

    def print_summary(self):
        self.feature_layer.print_summary()
        #self.summary()

    def inference_network(self, inputs):
        state = inputs

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

    def sample_generate(self):
        q_z = tfp.distributions.Normal(loc=np.zeros(self.atoms, dtype=np.float32),
                                       scale=np.ones(self.atoms, dtype=np.float32))
        z = q_z.sample([1])
        p_x_given_z_logits = self.decoder(z)
        # p_x_given_z = tfp.distributions.Bernoulli(logits=p_x_given_z_logits) # Bernoulli for binary image
        p_x_given_z = tfp.distributions.MultivariateNormalDiag(
                #tf.reshape(p_x_given_z_logits,[-1]), scale_identity_multiplier=0.05)
                self.flat(p_x_given_z_logits), scale_identity_multiplier=0.05)
        return p_x_given_z, p_x_given_z_logits

    def call(self, inputs):
        state = inputs

        # Encoder
        n_sample = state.shape[0]
        q_z, z_mean, z_log_var = self.inference_network(inputs)

        # Decoder
        p_x_given_z, z_decoded = self.generative_network(q_z)

        # Reconstruction Distribution
        z = z_mean
        #z = q_z.sample()

        # Critic
        #phi = self.phi_dense1(net)
        phi = z
        r_pred = self.successor_weight(phi)
        #psi = self.psi_dense1(tf.stop_gradient(z))
        psi = self.psi_dense1(tf.stop_gradient(phi))
        psi = self.psi_dense2(psi)
        critic = self.successor_weight(psi, training=False)

        # UNREAL-RP
        rp = self.rp_dense(z)

        feature = {'latent': z,
                   'z_mean': z_mean,
                   'z_log_var': z_log_var,
                   'z_decoded': z_decoded,
                   'q_z': q_z,
                   'p_x_given_z': p_x_given_z}
        SF = {'reward_predict': r_pred,
              'phi': phi,
              'psi': psi,
              'critic': critic,
              'UNREAL_rp': rp}

        return SF, feature 
    
@tf.function
def loss_reward_central(model, state, reward, training=True):
    # Critic - Reward Prediction
    inputs = state
    SF, feature = model(inputs, training=training)
    r_pred = SF['reward_predict']
    reward_mse = model.reward_loss(tf.cast(reward, tf.float32), r_pred)
    #reward_mse = tf.reduce_mean(tf.square(reward - r_pred))

    # UNREAL - RP
    rp = SF['UNREAL_rp']
    reward_label = reward + 1
    reward_label = tf.math.maximum(reward+1,0)
    reward_label = tf.math.minimum(reward+1,2)
    reward_label = tf.one_hot(tf.cast(reward_label, tf.int32), depth=3)
    rp_loss = model.rp_loss(reward_label, rp)

    total = 0.1*rp_loss#reward_mse + rp_loss

    info = {'reward_mse': reward_mse, 'rp_loss': rp_loss} 
    return total, info

#@tf.function
def loss_central_critic(model, state, td_target, td_target_c,
        psi_beta=1.0, beta_kl=1e-2, elbo_beta=1e-4, critic_beta=0.5,
        gamma=0.98, training=True):
    inputs = state
    SF, feature = model(inputs, training=training)

    # Critic - TD Difference
    psi = SF['psi']
    td_target = tf.cast(td_target, tf.float32)
    psi_mse = tf.reduce_mean(tf.square(td_target - psi))

    td_target_c = tf.cast(td_target_c, tf.float32)
    critic_mse = model.critic_loss(td_target_c, SF['critic'])

    # VAE ELBO loss
    p_z = tfp.distributions.Normal(loc=np.zeros(model.atoms, dtype=np.float32),
                                   scale=np.ones(model.atoms, dtype=np.float32))
    q_z = feature['q_z']
    p_x_given_z = feature['p_x_given_z']
    e_log_likelihood = p_x_given_z.log_prob(model.flat(state)) # Reconstruction term
    kl_loss = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, p_z), 1)
    elbo = tf.reduce_mean(e_log_likelihood - kl_loss, axis=0)

    total_loss = psi_beta*psi_mse - elbo_beta*elbo + critic_beta*critic_mse

    _, sample_generated_image = model.sample_generate()

    info = {
        'psi_mse': psi_mse,
        'elbo': -elbo,
        'critic_mse': critic_mse,
        'sample_generated_image': sample_generated_image[0],
        'sample_decoded_image': feature['z_decoded'].numpy()[0]
        }

    return total_loss, info

#@tf.function
def loss_ppo(model, state, belief, old_log_logit, action, td_target, advantage, td_target_c,
         eps=0.2, entropy_beta=0.3, psi_beta=0.1, decoder_beta=1e-4, gamma=0.98,
         training=True):
    num_sample = state.shape[0]

    # Run Model
    pi, v = model([state, belief])
    actor = pi['softmax']
    psi = v['psi']
    log_logits = pi['log_softmax']

    # Decoder loss
    generator_loss = tf.reduce_mean(tf.square(state - v['decoded_state']))

    # Entropy
    entropy = -tf.reduce_mean(actor * tf_log(actor), name='entropy')

    # Psi Loss
    td_target = tf.cast(td_target, tf.float32)
    #psi_mse = tf.reduce_mean(tf.square(td_target - psi))
    psi_mse = model.psi_loss_func(td_target, psi)

    critic_mse = model.psi_loss_func(td_target_c, v['critic'])

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

    total_loss = actor_loss + psi_beta*psi_mse - entropy_beta*entropy + decoder_beta*generator_loss + 0.5*critic_mse
    info = {'actor_loss': actor_loss,
            'psi_loss': psi_mse,
            'critic_mse': critic_mse,
            'entropy': entropy,
            'generator_loss': generator_loss}

    return total_loss, info

#@tf.function
def loss_multiagent_critic(model, states_list, belief, value_central, mask, training=True):
    # states_list and belief should be given in flat format
    num_env = mask.shape[0]
    num_agent = mask.shape[1]
    pi, v = model([states_list, belief], training=training)
    critics = tf.reshape(v['critic'], mask.shape) * tf.cast(mask, tf.float32)
    group_critic = tf.reduce_sum(critics, axis=1)
    #mse = tf.reduce_mean(tf.square(value_central - group_critic))
    mse = model.critic_loss_func(value_central, group_critic)
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

