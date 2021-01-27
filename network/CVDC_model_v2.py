import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.attention import Non_local_nn

from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

import numpy as np


class Decentral(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_space, atoms=128,
            prebuilt_layers=None, trainable=True,gru_units=128):
        super(Decentral, self).__init__()
        self.gru_units=gru_units
        print("----v2------")

        if prebuilt_layers is None:
            # Feature Encoding
            self.feature_layer = keras.Sequential([
                layers.Input(shape=input_shape),
                layers.TimeDistributed(layers.Dense(units=256, activation='elu')),
            ])

            self.gru = layers.GRU(units=gru_units, activation='elu',return_state=True)
            '''
            self.feature_layer2= keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Flatten(),
                layers.Dense(units=256, activation='elu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Dense(units=atoms, activation='elu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            ])
            self.pi_layer2 = keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Flatten(),
                layers.Dense(units=256, activation='elu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                layers.Dense(units=atoms, activation='elu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            ])
            '''

            # Decoder
            self.action_dense1 = layers.Dense(units=256, activation='elu')
            self.decoder_pre_dense1 = layers.Dense(units=256, activation='elu')
            self.decoder_dense1 = layers.Dense(units=256, activation='elu')
            self.decoder = keras.Sequential([
                layers.Dense(units=256, activation='elu'),
                layers.Dense(units=256, activation='elu'),
                layers.Dense(units=np.prod(input_shape[1:]), activation='linear'),
                layers.Reshape(input_shape[1:])
            ])

            # Phi
            self.phi_dense1 = layers.Dense(units=atoms, activation='elu')

            # Psi
            self.psi_dense1 = layers.Dense(units=atoms, activation='elu')
            self.psi_dense2 = layers.Dense(units=atoms, activation='elu')
        else:
            # Feature Encoding
            self.feature_layer = prebuilt_layers.feature_layer
            self.gru = prebuilt_layers.gru

            # Decoder
            self.action_dense1 = prebuilt_layers.action_dense1
            self.decoder_pre_dense1 = prebuilt_layers.decoder_pre_dense1
            self.decoder_dense1 = prebuilt_layers.decoder_dense1
            self.decoder = prebuilt_layers.decoder

            # Phi
            self.phi_dense1 = prebuilt_layers.phi_dense1

            # Psi
            self.psi_dense1 = prebuilt_layers.psi_dense1
            self.psi_dense2 = prebuilt_layers.psi_dense2

        # Critic weights
        self.sf_v_weight = layers.Dense(units=1, activation='linear', use_bias=False,)
        self.sf_q_weight = layers.Dense(units=action_space, activation='linear', use_bias=False,)

        # Actor
        self.actor_dense1 = layers.Dense(action_space)
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        self.smoothed_pseudo_H = tf.Variable(1.0)

        # Learnabilty Maximizer
        beta = np.ones([atoms,1], dtype=np.float32)
        self.beta = tf.Variable(
                initial_value=beta,
                name='feature_scale',
                dtype=tf.float32,
                constraint=tf.keras.constraints.MinMaxNorm(rate=0.99, axis=1),
            )

        # Loss
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.SUM)

        self._built = False
    def get_initial_state(self,batch_size):
        # a=tf.tile(tf.expand_dims(self.w,0),[batch_size,1])
        # print(self.w)
        # return a
        return np.zeros([batch_size,self.gru_units])

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs): # Full Operation of the method
        # Run full network
        obs = inputs[0]
        action = inputs[1] # Action is included for decoder
        avail_actions = tf.cast(inputs[2], tf.float32)
        initial_gru = tf.cast(inputs[3], tf.float32)
        num_sample = obs.shape[0]
        action_one_hot = tf.one_hot(action, self.action_space, dtype=tf.float32)

        # Feature Encoding SF-phi
        phi = self.feature_layer(obs)
        phi,final_state=self.gru(phi,initial_state=initial_gru)
        phi = self.phi_dense1(phi)
        #phi_norm = tf.norm(phi, ord=1, axis=1, keepdims=True)
        #phi = tf.math.divide_no_nan(phi, phi_norm)

        # Actor
        #net = tf.concat([net, tf.stop_gradient(phi)], axis=1)
        net = self.actor_dense1(phi)
        inf_mask = tf.maximum(tf.math.log(avail_actions), tf.float32.min)
        net = inf_mask + net
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

        #net = self.psi_dense1(phi)
        #net = self.psi_dense2(net)
        net = phi
        critic = self.sf_v_weight(net, training=True)
        critic = tf.reshape(critic, [-1])
        q = self.sf_q_weight(net, training=True)

        beta = tf.math.abs(self.beta)
        wv = self.sf_v_weight.weights[0]
        wq = self.sf_q_weight.weights[0]
        wv_neg = wv * (1.0-beta)
        reward_predict = tf.linalg.matmul(tf.stop_gradient(phi), wv_neg) # inverse approx
        reward_predict = tf.reshape(reward_predict, [-1])
        inv_critic = tf.linalg.matmul(psi, wv_neg)
        inv_critic = tf.reshape(inv_critic, [-1])

        # For learnability update
        wq_pos = tf.stop_gradient(wq) * beta
        wv_neg = tf.stop_gradient(wv) * (1.0-beta)
        psi_q_pos = tf.linalg.matmul(tf.stop_gradient(psi), wq_pos)
        psi_v_neg = tf.linalg.matmul(tf.stop_gradient(psi), wv_neg)

        # Filtered decoder
        _phi = phi*tf.transpose(beta)
        _dec_net = self.decoder_pre_dense1(_phi)
        _net = tf.math.multiply(_dec_net, act_net)
        _net = self.decoder_dense1(_net)
        _decoded_state = self.decoder(_net)

        actor = {'softmax': softmax_logits,
                 'final_state': final_state,
                 'log_softmax': log_logits}
        SF = {'reward_predict': reward_predict,
              'phi': phi,
              'psi': psi,
              'critic': critic,
              'decoded_state': decoded_state,
              'Q': q,
              'icritic': critic - 0.1*inv_critic, # Corrected critic
              'psi_q_pos': psi_q_pos,
              'psi_v_neg': psi_v_neg,
              'filtered_decoded_state': _decoded_state,
              }

        self._built = True

        return actor, SF

class Central(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, atoms,
                 trainable=True):
        super(Central, self).__init__()

        # Feature Encoding
        self.feature_layer = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.TimeDistributed(layers.Dense(units=256, activation='elu')),
            #layers.GRU(64),
            layers.TimeDistributed(layers.Dense(units=256, activation='elu')),
            layers.Flatten()
        ])

        # Critic
        self.critic_dense1 = layers.Dense(units=1, activation='linear')

        #self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, inputs):
        # Encoder
        net = self.feature_layer(inputs)
        net = self.critic_dense1(net)
        critic = tf.squeeze(net)

        feature = {'latent': net}
        SF = {'critic': critic}

        return SF, feature

@tf.function
def loss_central(model, state, td_target_c, old_value):
    eps = 0.2
    inputs = state
    SF, feature = model(inputs)

    # Critic - TD Difference
    v_pred = SF['critic']
    v_pred_clipped = old_value + tf.clip_by_value(v_pred-old_value, -eps, eps)
    critic_mse = tf.minimum(
        tf.square(v_pred - td_target_c),
        tf.square(v_pred_clipped - td_target_c))
    critic_mse = tf.reduce_mean(critic_mse)

    total_loss = critic_mse
    # print(critic_mse)
    info = {'critic_mse': critic_mse}

    return total_loss, info

@tf.function
def loss_ppo(model, state, old_log_logit, action, old_value, td_target, advantage, td_target_c, rewards, next_state, avail_actions,
        eps, entropy_beta, q_beta, psi_beta, decoder_beta, critic_beta, learnability_beta, reward_beta,initial_state,final_state):
    num_sample = state.shape[0]

    # Run Model
    pi, v = model([state, action, avail_actions,initial_state])
    pi_next, v_next = model([next_state, action, avail_actions,final_state]) # dummy action
    actor = pi['softmax']
    psi = v['psi']
    log_logits = pi['log_softmax']

    # Reward Accuracy
    reward_loss = model.mse_loss_sum(rewards, v['reward_predict'])
    #reward_loss = 0.0

    # Decoder loss
    generator_loss = model.mse_loss_sum(next_state[:,-1,:], v['decoded_state'])
    #generator_loss = 0.0

    # Entropy
    H = -tf.reduce_mean(actor * tf_log(actor), axis=-1) # Entropy H of each sample
    mean_entropy = tf.reduce_sum(H)
    pseudo_H = tf.stop_gradient(
            tf.reduce_sum(actor*(1-actor), axis=-1))
    mean_pseudo_H = tf.reduce_mean(pseudo_H)
    smoothed_pseudo_H = model.smoothed_pseudo_H

    #Adaptive_Entropy from LICA
    num_actions = actor.shape[1]
    a_acts = tf.cast(avail_actions, tf.float32)

    adaptive_entropy = (tf_log(actor)+1.0)/tf.clip_by_value(tf.tile(tf.stop_gradient(tf.expand_dims(H,axis=-1)),tf.constant([1,num_actions], tf.int32)),0.00001,1)*a_acts
    # exit()
    adaptive_entropy = tf.reduce_sum(adaptive_entropy)

    # KL Divergence

    # Critic Loss
    v_pred = v['critic']
    v_pred_clipped = old_value + tf.clip_by_value(v_pred-old_value, -eps, eps)
    critic_mse = tf.minimum(
        tf.square(v_pred - td_target_c),
        tf.square(v_pred_clipped - td_target_c))
    critic_mse = tf.reduce_sum(critic_mse)
    #critic_mse = tf.reduce_mean(critic_mse * tf.stop_gradient(smoothed_pseudo_H)) + tf.square(mean_pseudo_H-smoothed_pseudo_H)
    #critic_mse = tf.reduce_mean(tf.square(v_pred-td_target_c))

    # Psi Loss
    psi_mse = model.mse_loss_mean(td_target, psi)
    #psi_mse = 0.0

    # Actor Loss
    action_one_hot = tf.one_hot(action, model.action_space, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_one_hot, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_one_hot, 1)
    ratio = tf.exp(log_prob - old_log_prob) # precision: log_prob / old_log_prob
    #ratio = tf.clip_by_value(ratio, -1e8, 1e8)
    surrogate = ratio * advantage # Clipped surrogate function
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate)
    actor_loss = -tf.reduce_sum(surrogate_loss)

    # KL
    approx_kl = tf.reduce_mean(old_log_prob - log_prob)
    approx_ent = tf.reduce_mean(-log_prob)

    # Q - Loss
    q = v['Q']
    #q_a = tf.reduce_sum(q * tf.stop_gradient(actor), 1)
    q_a = tf.reduce_sum(q * action_one_hot, 1)  # Current Q value
    q_target = tf.stop_gradient(rewards + 0.98 * tf.reduce_max(v_next['Q'], axis=1))
    #q_loss = tf.reduce_mean(tf.square(td_target_c - q_a))
    q_loss = tf.reduce_mean(tf.square(q_target - q_a))

    # L2 loss
    #l2_loss = tf.nn.l2_loss(model.sf_v_weight.weights[0]) + tf.nn.l2_loss(model.sf_q_weight.weights[0])

    # Learnability
    var_action = tf.reduce_sum(tf.square(v['psi_q_pos']-v_next['psi_q_pos']) * tf.stop_gradient(pi['softmax']), axis=1)
    var_environment = tf.square((v['psi_v_neg'] - v_next['psi_v_neg'])[:,0])
    #learnability_loss = tf.reduce_mean(-var_action)
    learnability_loss = tf.reduce_mean(-var_action+0.5*var_environment)

    total_loss = actor_loss
    total_loss += psi_beta*psi_mse
    total_loss += entropy_beta*(-mean_entropy)
    total_loss += decoder_beta*generator_loss
    total_loss += reward_beta*reward_loss
    total_loss += critic_beta*critic_mse
    total_loss += q_beta*q_loss
    total_loss += learnability_beta*learnability_loss
    #total_loss += 0.001*l2_loss

    # Log
    info = {'actor_loss': actor_loss,
            'psi_loss': psi_mse,
            'critic_mse': critic_mse,
            'entropy': mean_entropy,
            'adaptive_entropy': adaptive_entropy,
            'generator_loss': generator_loss,
            'q_loss': q_loss,
            'reward_loss': reward_loss,
            'learnability_loss': learnability_loss,
            'approx_kl': approx_kl,
            'approx_ent': approx_ent,
            }

    return total_loss, info

'''
def get_gradient(model, loss, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        loss_val, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(loss_val, model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return grads, info
'''

def train(model, loss, optimizer, inputs, hyperparameters={}):
    with tf.GradientTape() as tape:
        total_loss, info = loss(model, **inputs, **hyperparameters)
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, grad_norm = tf.clip_by_global_norm(grads, 10)
    info["grad_norm"] = grad_norm
    optimizer.apply_gradients([
        (
            # tf.clip_by_norm(grad, 5.0),
            #tf.clip_by_value(grad, -5.0, 5.0),
            grad,
            var
        )
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])
    return total_loss, info
