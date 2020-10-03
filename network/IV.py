import os
import sys
import math

from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

from network.model_V4_30 import V4
from utility.utils import store_args
from utility.tf_utils import tf_clipped_log as tf_log

import numpy as np

def loss_ppo(model, state, old_log_logit, action, advantage, 
         eps=0.2, entropy_beta=0.05, return_losses=False):
    num_sample = state.shape[0]

    # Run Model
    actor, critic, log_logits = model(state)

    # Entropy
    H = -tf.reduce_mean(actor * tf_log(actor), axis=-1)
    mean_entropy = tf.reduce_mean(H)
    pseudo_H = tf.stop_gradient(
            tf.reduce_sum(actor*(1-actor), axis=-1))
    mean_pseudo_H = tf.reduce_mean(pseudo_H)
    smoothed_pseudo_H = model.smoothed_pseudo_H

    # Actor Loss
    action_OH = tf.one_hot(action, model.action_size, dtype=tf.float32)
    log_prob = tf.reduce_sum(log_logits * action_OH, 1)
    old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

    # Clipped surrogate function
    ratio = tf.exp(log_prob - old_log_prob) # precision
    #ratio = log_prob / old_log_prob
    surrogate = ratio * advantage
    clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
    surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
    actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

    total_loss = actor_loss - entropy_beta*mean_entropy

    info = None
    if return_losses:
        info = {'actor_loss': actor_loss,
                'entropy': mean_entropy}

    return total_loss, info

def loss_critic(models, state, td_target, old_value, agent_type_index,
        eps=0.2, critic_beta=0.5, return_losses=False):
    num_sample = state.shape[0]

    # Run Model
    central_critic = None
    for idx, atype in enumerate(agent_type_index):
        model = models[atype]
        _, critic, _= model(state[:,idx,...])
        if central_critic is None:
            central_critic = critic
        else:
            central_critic += critic

    # Critic Loss
    critic_loss = tf.reduce_mean(tf.square(central_critic - td_target))

    total_loss = 0.5*critic_beta*critic_loss

    info = None
    if return_losses:
        info = {'critic_loss': critic_loss}

    return total_loss, info

def train_actor(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        total_loss, info = loss_ppo(model, **inputs, **hyperparameters, return_losses=True)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, model.trainable_variables)
        if grad is not None])

    return total_loss, info

def train_critic(model, optimizer, inputs, **hyperparameters):
    with tf.GradientTape() as tape:
        total_loss, info = loss_critic(model, **inputs, **hyperparameters, return_losses=True)
    variables = []
    for t in model:
        variables += t.trainable_variables
    grads = tape.gradient(total_loss, variables)
    optimizer.apply_gradients([
        (grad, var)
        for (grad,var) in zip(grads, variables)
        if grad is not None])

    return total_loss, info

