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

        # Critic
        self.critic_dist = layers.Dense(units=atoms, activation='softmax')

        # Critic Loss
        self.zloss = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):
        net = self.feature_network(inputs)

        critic_dist = self.critic_dist(net)
        critic = tf.reduce_sum(critic_dist * self.z, axis=1)

        return critic, critic_dist

def loss(model, target_model, state, reward, done, next_state,
        gamma=0.98, training=True):
    num_sample = state.shape[0]

    # Run Model
    v_out, z_out  = model(state, training=training)
    v_next, z_next = model(next_state)
    #_,_,v_next_targ, z_next_targ = target_model(next_state) # target network

    # Critic Loss
    # Project Next State Value Distribution (of optimal action) to Current State
    m_prob = np.zeros((num_sample, model.atoms), dtype=np.float32)
    Tz = tf.minimum(model.v_max, tf.maximum(model.v_min, reward[:,None] + gamma * model.z * (1-done[:,None])))
    bj = (Tz - model.v_min) / model.delta_z 
    m_l, m_u = tf.math.floor(bj+1e-6).numpy().astype(int), tf.math.ceil(bj-1e-6).numpy().astype(int)
    A = z_next * (m_u - bj)
    B = z_next * (bj - m_l)
    for j in range(model.atoms):
        #m_prob[:,m_l] += (z_next_targ[:,j]**(1-done)) * (m_u - bj)
        #m_prob[:,m_u] += (z_next_targ[:,j]**(1-done)) * (bj - m_l)
        #m_prob[:,m_l] += (z_next[:,j]**(1-done)) * (m_u - bj)
        #m_prob[:,m_u] += (z_next[:,j]**(1-done)) * (bj - m_l)
        m_prob[:,m_l[:,j]] += A[:,j]
        m_prob[:,m_u[:,j]] += B[:,j]

    #critic_loss = -tf.reduce_sum(m_prob * tf.math.log(z_out), axis=-1)
    critic_loss = model.zloss(m_prob, z_out) # Same as cross-entropy loss

    return critic_loss

def train(model, target_model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        loss_val = loss(model, target_model, **inputs, **hyperparameters, training=True)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_val

