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
            trainable=True):
        super(V4SF_CVDC_DECENTRAL, self).__init__()

        # Feature Encoding
        self.feature_layer = V4Decentral(input_shape, action_size, name='feature_encoding')
        self.pi_layer = V4Decentral(input_shape, action_size, name='pi_encoding')

        # Decoder
        self.action_dense1 = layers.Dense(units=128, activation='relu')
        self.decoder_pre_dense1 = layers.Dense(units=128, activation='relu')
        self.decoder_dense1 = layers.Dense(units=128, activation='relu')
        self.decoder = V4INVDecentral()

        # Phi
        self.phi_dense1 = layers.Dense(units=atoms, activation='relu')
        self.sf_v_weight = layers.Dense(units=1, activation='linear', use_bias=False,)
                #kernel_constraint=tf.keras.constraints.MaxNorm(2))
        self.sf_q_weight = layers.Dense(units=action_size, activation='linear', use_bias=False,)
                #kernel_constraint=tf.keras.constraints.MaxNorm(2))

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size, activation='relu')
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Psi
        self.psi_dense1 = layers.Dense(units=128, activation='relu')
        self.psi_dense2 = layers.Dense(units=atoms, activation='relu')
        self.smoothed_pseudo_H = tf.Variable(1.0)

        # Learnabilty Maximizer
        beta = np.ones([atoms,1], dtype=np.float32)
        self.beta = tf.Variable(
                initial_value=beta,
                name='feature_scale',
                dtype=tf.float32,
                constraint=[tf.keras.constraints.MinMaxNorm(rate=0.99, axis=1),
                            tf.keras.constraints.NonNeg()],
            )

        # Loss
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Run full network
        obs = inputs[0]
        action = inputs[1] # Action is included for decoder
        action_one_hot = tf.one_hot(action, self.action_size, dtype=tf.float32)

        # Feature Encoding SF-phi
        net = self.feature_layer(obs)
        phi = self.phi_dense1(net)
        phi = phi / tf.norm(phi, ord=1, axis=1, keepdims=True)

        # Actor
        net = self.pi_layer(obs)
        net = tf.concat([net, tf.stop_gradient(phi)], axis=1)
        net = self.actor_dense1(net)
        net = self.actor_dense2(net)
        softmax_logits = self.softmax(net)
        log_logits = self.log_softmax(net)

        # Decoder
        dec_net = self.decoder_pre_dense1(phi)
        act_net = self.action_dense1(action_one_hot)
        net = tf.math.multiply(dec_net, act_net)
        net = self.decoder_dense1(net)
        decoded_state = self.decoder(net)

        psi = self.psi_dense1(tf.stop_gradient(phi))
        psi = self.psi_dense2(psi)

        net = self.psi_dense1(phi)
        net = self.psi_dense2(net)
        critic = self.sf_v_weight(net, training=True)
        q = self.sf_q_weight(net, training=True)

        # Masking method for lowest 1/8th variation
        '''
        q_w = self.sf_q_weight.weights[0]
        q_w_std = tf.math.reduce_std(q_w, axis=1, keepdims=True)
        w_mask = tf.cast(q_w_std[tf.argsort(q_w_std, axis=0)[-8,0]] <= q_w_std, tf.float32)
        w = self.sf_v_weight.weights[0]
        w = w * w_mask
        #reward_predict = self.sf_v_weight(phi, training=False)
        reward_predict = tf.linalg.matmul(phi, w)
        inv_critic = tf.linalg.matmul(psi, w)
        '''

        wv = self.sf_v_weight.weights[0]
        wq = self.sf_q_weight.weights[0]
        wv_neg = wv * (1.0-self.beta)
        reward_predict = tf.linalg.matmul(phi, wv_neg)
        inv_critic = tf.linalg.matmul(psi, wv_neg)

        # For learnability
        wq_pos = tf.stop_gradient(wq) * self.beta
        wv_neg = tf.stop_gradient(wv) * (1.0-self.beta)
        psi_q_pos = tf.linalg.matmul(tf.stop_gradient(psi), wq_pos)
        psi_v_neg = tf.linalg.matmul(tf.stop_gradient(psi), wv_neg)

        actor = {'softmax': softmax_logits,
                 'log_softmax': log_logits}
        SF = {'reward_predict': reward_predict,
              'phi': phi,
              'psi': psi,
              'critic': critic,
              'decoded_state': decoded_state,
              'Q': q,
              'icritic': critic - inv_critic,
              'psi_q_pos': psi_q_pos,
              'psi_v_neg': psi_v_neg,
              }

        return actor, SF


class V4SF_CVDC_CENTRAL(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, atoms,
                 trainable=True):
        super(V4SF_CVDC_CENTRAL, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape)

        # Decoding
        self.decoder = V4INV()

        # Phi
        self.phi = layers.Dense(units=atoms, activation='sigmoid')
        #self.phi_dense1 = layers.Dense(units=atoms, activation='elu')
        self.successor_weight = layers.Dense(units=1, activation='linear', use_bias=False,
                kernel_constraint=tf.keras.constraints.MaxNorm(2))
                #kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        # Psi
        self.psi_dense1 = layers.Dense(units=64, activation='elu')
        self.psi_dense2 = layers.Dense(units=atoms, activation='elu')
        self.psi_dropout = layers.Dropout(0.3)

        # UNREAL-RP
        self.rp_dense = layers.Dense(units=3, activation='softmax')

        # Misc Layers
        self.flat = layers.Flatten()

        # Loss
        
        # Loss Operations
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)
        self.rp_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    def print_summary(self):
        self.feature_layer.print_summary()
        #self.summary()

    def call(self, inputs):
        state = inputs

        # Encoder
        n_sample = state.shape[0]

        # Phi
        net = self.feature_layer(state)
        net = self.phi(net)

        # Decoder
        z_decoded = self.decoder(net)

        # Critic
        #phi = self.phi_dense1(net)
        phi = net
        r_pred = self.successor_weight(net, training=False)

        psi = self.psi_dense1(tf.stop_gradient(net))
        psi = self.psi_dense2(psi)
        psi = self.psi_dropout(psi)
        critic = self.successor_weight(psi, training=True)

        # UNREAL-RP
        rp = self.rp_dense(net)

        feature = {'latent': net,
                   'z_decoded': z_decoded,}
        SF = {'phi': phi,
              'reward_predict': r_pred,
              'psi': psi,
              'critic': critic,
              'UNREAL_rp': rp,}

        return SF, feature 

@tf.function
def loss_central(model, state, td_target, td_target_c, reward,
        psi_beta, beta_kl, recon_beta, critic_beta, unreal_rp_beta):
    inputs = state
    SF, feature = model(inputs)

    # Reward Prediction Accuracy (regression, not trained)
    pred_reward = SF['reward_predict']
    reward_acc = model.mse_loss_sum(reward, pred_reward)

    # UNREAL - reward prediction
    rp = SF['UNREAL_rp']
    reward_label = reward + 1
    reward_label = tf.math.maximum(reward+1,0)
    reward_label = tf.math.minimum(reward+1,2)
    reward_label = tf.one_hot(reward_label, depth=3)
    rp_loss = model.rp_loss(reward_label, rp)

    # Critic - TD Difference
    psi = SF['psi']
    td_target = tf.cast(td_target, tf.float32)
    psi_mse = tf.reduce_mean(tf.square(td_target - psi))

    td_target_c = tf.cast(td_target_c, tf.float32)
    critic_mse = model.mse_loss_mean(td_target_c, SF['critic'])

    # Reconstruction loss
    z_decoded = feature['z_decoded']
    recon_loss = model.mse_loss_sum(state, z_decoded)

    total_loss = psi_beta*psi_mse - recon_beta*recon_loss+ critic_beta*critic_mse + unreal_rp_beta*rp_loss

    info = {
        'psi_mse': psi_mse,
        'elbo': -elbo,
        'critic_mse': critic_mse,
        'sample_decoded_image': z_decoded[0],
        'reward_mse': reward_acc,
        'rp_loss': rp_loss
        }

    return total_loss, info

@tf.function
def loss_ppo(model, state, old_log_logit, action, old_value, td_target, advantage, td_target_c, rewards, next_state,
        eps, entropy_beta, q_beta, psi_beta, decoder_beta, critic_beta, learnability_beta,):
    num_sample = state.shape[0]

    # Run Model
    pi, v = model([state, action])
    pi_next, v_next = model([next_state, action])
    actor = pi['softmax']
    psi = v['psi']
    log_logits = pi['log_softmax']

    # Reward Accuracy
    reward_loss = model.mse_loss_sum(rewards, v['reward_predict'])

    # Decoder loss
    #generator_loss = model.mse_loss_sum(state, v['decoded_state'])
    generator_loss = model.mse_loss_sum(next_state, v['decoded_state'])

    # Entropy
    H = -tf.reduce_sum(actor * tf_log(actor), axis=-1) # Entropy H of each sample
    mean_entropy = tf.reduce_mean(H)
    pseudo_H = tf.stop_gradient(
            tf.reduce_sum(actor*(1-actor), axis=-1))
    mean_pseudo_H = tf.reduce_mean(pseudo_H)
    smoothed_pseudo_H = model.smoothed_pseudo_H

    # Critic Loss
    v_pred = v['critic']
    v_pred_clipped = old_value + tf.clip_by_value(v_pred-old_value, -eps, eps)
    critic_mse = tf.maximum(
        tf.square(v_pred - td_target_c),
        tf.square(v_pred_clipped - td_target_c))
    critic_mse = tf.reduce_mean(critic_mse * tf.stop_gradient(smoothed_pseudo_H)) + tf.square(mean_pseudo_H-smoothed_pseudo_H)
    
    # Psi Loss
    psi_mse = model.mse_loss_mean(td_target, psi)

    # Actor Loss
    action_one_hot = tf.one_hot(action, model.action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_one_hot, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_one_hot, 1)
    ratio = tf.exp(log_prob - old_log_prob) # precision: log_prob / old_log_prob
    surrogate = ratio * advantage # Clipped surrogate function
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    # Q - Loss
    q = v['Q']
    q_a = tf.reduce_sum(q * tf.stop_gradient(actor), 1)
    q_loss = tf.reduce_mean(tf.square(q_a - td_target_c))

    # L2 loss
    #l2_loss = tf.nn.l2_loss(model.sf_v_weight.weights[0]) + tf.nn.l2_loss(model.sf_q_weight.weights[0])

    # Learnability
    var_action = tf.reduce_sum(tf.square(v['psi_q_pos']-v_next['psi_q_pos']) * tf.stop_gradient(pi['softmax']), axis=1)
    var_environment = tf.square((v['psi_v_neg'] - v_next['psi_v_neg'])[:,0])
    #learnability_loss = tf.reduce_mean(-var_action)
    learnability_loss = tf.reduce_mean(-var_action+0.5*var_environment)

    total_loss = actor_loss
    total_loss += psi_beta*psi_mse
    total_loss -= entropy_beta*mean_entropy
    total_loss += decoder_beta*generator_loss
    total_loss += critic_beta*critic_mse
    total_loss += q_beta*q_loss
    total_loss += learnability_beta*learnability_loss
    #total_loss += 0.001*l2_loss

    # Log
    info = {'actor_loss': actor_loss,
            'psi_loss': psi_mse,
            'critic_mse': critic_mse,
            'entropy': mean_entropy,
            'generator_loss': generator_loss,
            'q_loss': q_loss,
            'reward_loss': reward_loss,
            'learnability_loss': learnability_loss,
            }

    return total_loss, info

@tf.function
def loss_multiagent_critic(model, team_state, value_central, rewards, mask):
    # states_list should be given in flat format
    num_batch, num_agent, lx, ly, lc = team_state.shape
    mask = tf.cast(mask, tf.float32)

    states_list = tf.reshape(team_state, [num_batch*num_agent, lx, ly, lc])
    _, v = model(states_list)
    critics = tf.reshape(v['critic'], mask.shape) * mask
    group_critic = tf.reduce_sum(critics, axis=1)
    mse = model.mse_loss_mean(value_central, group_critic)

    rewards_prediction = tf.reshape(v['reward_predict'], mask.shape) * mask
    rewards_prediction = tf.reduce_sum(rewards_prediction, axis=1)
    reward_loss = model.mse_loss_sum(rewards, rewards_prediction)

    total_loss = 0.001*mse
    info = {'ma_critic': mse, 'reward_loss': reward_loss}
    return total_loss, info

def get_gradient(model, loss, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return grads, info

def train(model, loss, optimizer, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val, info

