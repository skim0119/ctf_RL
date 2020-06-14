import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from network.attention import Non_local_nn
from utility.utils import store_args

import numpy as np

class V2Dist(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5,
                 v_max=10, v_min=-10, atoms=51,
                 trainable=True, name='PPO'):
        super(V2Dist, self).__init__(name=name)

        self.delta_z = (self.v_max-self.v_min)/(self.atoms-1.0)
        self.z = tf.constant(self.v_min + self.delta_z*np.arange(self.atoms), dtype=tf.float32, shape=(1,atoms))

        # Feature Encoder
        self.feature_network = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.SeparableConv2D(
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    padding='valid',
                    depth_multiplier=16,
                    activation='relu',
                ),
            Non_local_nn(16),
            layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
            layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(units=128, activation='relu')])

        # Actor
        self.actor_dense1 = layers.Dense(action_size)

        # Critic
        self.critic_dist = keras.Sequential([
            layers.Dense(units=atoms, activation='softmax')])

        # Loss
        self.zloss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):
        net = self.feature_network(inputs)

        logits = self.actor_dense1(net) 
        actor = tf.nn.softmax(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic_dist = self.critic_dist(net)
        critic = tf.reduce_sum(critic_dist * self.z, axis=1)

        return actor, log_logits, critic, critic_dist

def _log(val):
    return tf.math.log(tf.clip_by_value(val, 1e-10, 10.0))

def get_action(model, state):
    actor, log_logit, critic, critic_dist = model(state)
    action_size = actor.shape[1]
    actor_prob = actor.numpy() / actor.numpy().sum(axis=1, keepdims=True)
    action = np.array([np.random.choice(action_size, p=prob) for prob in actor_prob])
    return action, critic, log_logit

def loss(model, state, action, td_target, advantage, old_log_logit, reward, done, next_state,
        action_size=5, eps=0.2, beta_entropy=0.05, beta_critic=0.5, gamma=0.98,
        training=True, return_losses=False):
    num_sample = state.shape[0]

    # Run Model
    actor, log_logit, v_out, z_out  = model(state, training=training)
    _, _, v_next, z_next = model(next_state)
    #_,_,v_next_targ, z_next_targ = self.target_model(next_state) # target network

    # Entropy
    entropy = -tf.reduce_mean(actor * _log(actor), name='entropy')

    # Critic Loss
    # Project Next State Value Distribution (of optimal action) to Current State
    m_prob = np.zeros((num_sample, model.atoms))
    for j in range(model.atoms):
        Tz = tf.minimum(model.v_max, tf.maximum(model.v_min, reward + gamma * model.z[0,j] * (1-done)))
        bj = (Tz - model.v_min) / model.delta_z 
        m_l, m_u = tf.math.floor(bj).numpy().astype(int), tf.math.ceil(bj).numpy().astype(int)
        m_prob[:,m_l] += (z_next[:,j]**(1-done)) * (m_u - bj)
        m_prob[:,m_u] += (z_next[:,j]**(1-done)) * (bj - m_l)
    critic_loss = model.zloss(m_prob, z_out)
    #td_error = td_target - critic
    #critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

    # Actor Loss
    action_OH = tf.one_hot(action, action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logit * action_OH, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

    # Clipped surrogate function
    ratio = tf.exp(log_prob - old_log_prob) # precision
    #ratio = log_prob / old_log_prob
    surrogate = ratio * advantage
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    total_loss = actor_loss + (beta_critic * critic_loss) - (beta_entropy * entropy)
    if return_losses:
        return total_loss, actor_loss, beta_critic*critic_loss, beta_entropy*entropy
    else:
        return total_loss

def train(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val = loss(model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_val

