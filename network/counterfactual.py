import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.model_V4_30 import V4, V4INV
from network.model_V4_30 import V4Decentral, V4INVDecentral

from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

import numpy as np


class V4COMA_d(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_size=5, atoms=128,
            trainable=True):
        super(V4COMA_d, self).__init__()

        # Feature Encoding
        self.feature_layer = V4(input_shape, action_size)

        # Actor
        self.actor_dense1 = layers.Dense(128, activation='relu')
        self.actor_dense2 = layers.Dense(action_size, activation='relu')
        self.softmax = layers.Activation('softmax')
        self.log_softmax = layers.Activation(tf.nn.log_softmax)

        # Loss
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()
        #self.mse_loss_sum = tf.keras.losses.MeanSquaredError(
        #        reduction=tf.keras.losses.Reduction.SUM)

    def print_summary(self):
        self.feature_layer.summary()

    def call(self, inputs):
        # Actor
        net = self.feature_layer(inputs)
        net = self.actor_dense1(net)
        net = self.actor_dense2(net)
        softmax_logits = self.softmax(net)
        log_logits = self.log_softmax(net)
        action = tf.squeeze(tf.random.categorical(log_logits, 1, dtype=tf.int32))

        actor = {'softmax': softmax_logits, 'log_softmax': log_logits, 'action': action}

        return actor


class V4COMA_c(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, indiv_models, agent_type_index, num_agent_type, action_size=5, atoms=128,
                 trainable=True):
        super(V4COMA_c, self).__init__()

        # Share feature layer 
        self.feature_layer = V4(input_shape, action_size)
        self.feature_dense1 = layers.Dense(units=128, activation='relu')

        # Action net
        self.action_dense1 = layers.Dense(units=8, activation='softmax')

        # Critic
        self.critic_dense1 = layers.Dense(units=atoms, activation='relu')
        #self.critic_dense2 = layers.Dense(units=atoms, activation='relu')
        self.critic_layer = layers.Dense(units=action_size, activation='linear')

        # Loss Operations
        self.mse_loss_mean = tf.keras.losses.MeanSquaredError()

    def call(self, env_state, indiv_states, indiv_actions):
        # dim[env_state] : [num_batch, ex, ey, ch]
        # dim[indiv_states] : [num_batch, num_agent, lx, ly, ch]
        # dim[indiv_actions] : [num_batch, num_agent]
        indiv_states = tf.unstack(indiv_states, axis=1)
        indiv_actions = tf.unstack(indiv_actions, axis=1)

        qvals = []
        env_net = self.feature_layer(env_state)
        for state, action, model in zip(indiv_states, indiv_actions, self.indiv_models):
            # Feature
            feature_net = model.feature_layer(state)
            #feature_net = tf.stop_gradient(feature_net)
            feature_net = self.feature_dense1(feature_net)
            # Action
            action_onehot = tf.one_hot(action, 5)
            action_net = self.action_dense1(action_onehot)
            # Critic  
            net = tf.concat([env_net, feature_net, action_net], axis=1)
            net = self.critic_dense1(net)
            qval = self.critic_layer(net)
            qvals.append(qval)

        return qvals


@tf.function
def loss(cent_model, dec_models, env_states, metastates, metaactions, rewards):
    critic_loss = None
    actor_loss = None
    qvals = cent_model(env_states, metastates, metaactions)
    for idx, qval in enumerate(qvals):
        # Centralized Network Loss
        qmax = tf.math.reduce_max(qval, axis=1)
        action = metaactions[:,idx]
        action_onehot = tf.one_hot(action, 5)
        q_a = tf.reduce_sum(qval * action_onehot, axis=1)
        qmax = tf.concat([qmax, [0]], axis=0)
        td_target = rewards + 0.98 * qmax[1:]
        mse = cent_model.mse_loss_mean(q_a, tf.stop_gradient(td_target))
        if critic_loss is None:
            critic_loss = mse
        else:
            critic_loss += mse

        # Decentralized Netowork Loss
        model = dec_models[idx]
        pi = model(metastates[:,idx,...])['softmax']
        counter = tf.one_hot(action, 5, dtype=tf.float32) - pi
        advantage = tf.stop_gradient(tf.reduce_sum(qval * counter, axis=1))
        
        pi_a = tf.reduce_sum(pi * action_onehot, axis=1)
        pg_loss = -tf.reduce_sum(advantage * tf_log(pi_a))
        if actor_loss is None:
            actor_loss = pg_loss
        else:
            actor_loss += pg_loss

    total_loss = actor_loss + critic_loss*0.5
    info = {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
    }

    return total_loss, info

def train(cent_model, dec_models, optimizer, inputs):
    with tf.GradientTape() as tape:
        total_loss, info = loss(cent_model, dec_models, **inputs)

    variables = cent_model.trainable_variables
    for t in dec_models:
        variables += t.trainable_variables
    # train central
    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, variables)
        if grad is not None])
    
    return total_loss, info

