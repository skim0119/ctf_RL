"""
Attention + sepCNN network for CtF encoder (June 11)

Mainly used for:
    A3C
    PPO
    VAE
"""

import os
import sys
sys.path.append('/home/namsong/github/raide_rl')

from functools import partial

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.keras.layers as keras_layers

from network.attention import non_local_nn_2d
from network.attention import Non_local_nn
from utility.utils import store_args

from method.base import put_channels_on_grid

class V2(tf.keras.Model):

    @store_args
    def __init__(self, trainable=True, name='V2'):
        super(V2, self).__init__(name=name)

        self.sep_conv2d = keras_layers.SeparableConv2D(
                filters=32,
                kernel_size=4,
                strides=2,
                padding='valid',
                depth_multiplier=4,
                activation='relu',
            )
        self.non_local = Non_local_nn(16)
        self.conv1 = keras_layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')
        self.conv2 = keras_layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu')
        self.flat  = keras_layers.Flatten()
        self.dense1 = keras_layers.Dense(units=128)

    def call(self, inputs):
        net = inputs
        _layers = {'input': net}

        # Block 1 : Separable CNN
        net = self.sep_conv2d(net)
        _layers['sepCNN1'] = net

        # Block 2 : Attention (with residual connection)
        net = self.non_local(net)
        _layers['attention'] = self.non_local._attention_map
        _layers['NLNN'] = net

        # Block 3 : Convolution
        net = self.conv1(net)
        _layers['CNN1'] = net
        net = self.conv2(net)
        _layers['CNN2'] = net

        # Block 4 : Feature Vector
        net = self.flat(net)
        _layers['flat'] = net
        net = self.dense1(net)
        _layers['dense1'] = net

        self._layers_snapshot = _layers

        if self.trainable:
            return net
        else:
            return tf.stop_gradient(net)


class V2_PPO(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, critic_beta=0.5, name='PPO'):
        super(V2_PPO, self).__init__(name=name)

        # Feature Encoder
        self.feature_network = V2()

        # Actor
        self.actor_dense1 = keras_layers.Dense(action_size)
        self.sftmx = keras_layers.Activation('softmax')

        # Critic
        self.critic_dense1 = keras_layers.Dense(1)

    def call(self, inputs):
        net = self.feature_network(inputs)

        logits = self.actor_dense1(net)
        logits = tf.math.maximum(logits, 1e-9)
        actor = self.sftmx(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic = self.critic_dense1(net)
        critic = tf.reshape(critic, [-1])

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic

        return actor, logits, log_logits, critic

    def build_loss(self, old_log_logit, action, advantage, td_target):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

            # Critic Loss
            td_error = td_target - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            total_loss = actor_loss
            if self.entropy_beta != 0:
                total_loss = actor_loss - entropy * self.entropy_beta
            if self.critic_beta != 0:
                total_loss = actor_loss + critic_loss * self.critic_beta

            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy

        return total_loss

    def _kl_entropy(self):
        # NOT FINISHED
        with tf.name_scope('kl_divergence'):
            target_logits = self.global_network.logits

            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
            a1 = target_logits - tf.reduce_max(target_logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, keepdims=True)
            z1 = tf.reduce_sum(ea1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)))

class V2_PPO_Termination(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, critic_beta=0.5, name='PPO'):
        super(V2_PPO_Termination, self).__init__(name=name)

        # Feature Encoder
        self.feature_network = V2()

        # Actor
        self.actor_dense1 = keras_layers.Dense(action_size)
        self.sftmx = keras_layers.Activation('softmax')

        # Critic
        self.critic_dense1 = keras_layers.Dense(1)

        # Termination
        self.term_dense1 = keras_layers.Dense(2)

    def call(self, inputs):
        net = self.feature_network(inputs)

        logits = self.actor_dense1(net)
        logits = tf.math.maximum(logits, 1e-9)
        actor = self.sftmx(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic = self.critic_dense1(net)
        critic = tf.reshape(critic, [-1])

        term_logits = self.term_dense1(net)
        term_logits = tf.math.maximum(term_logits, 1e-9)
        termination = self.sftmx(term_logits)

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.termination = termination

        return actor, logits, log_logits, critic, termination

    def build_loss(self, old_log_logit, action, advantage, td_target, termination, old_term_logit):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

            # Critic Loss
            td_error = td_target - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            #Termination Loss
            term_entropy = -tf.reduce_mean(self.termination * _log(self.termination), name='entropy_termination')
            term_OH = tf.one_hot(termination, 2, dtype=tf.float32)
            term_log_prob = tf.reduce_sum(self.termination * term_OH, 1)
            term_old_log_prob = tf.reduce_sum(old_term_logit * term_OH, 1)

            # Clipped surrogate function
            term_ratio = tf.exp(term_log_prob - term_old_log_prob)
            #ratio = log_prob / old_log_prob
            term_surrogate = term_ratio * advantage
            term_clipped_surrogate = tf.clip_by_value(term_ratio, 1-self.eps, 1+self.eps) * advantage
            term_surrogate_loss = tf.minimum(term_surrogate, term_clipped_surrogate, name='term_surrogate_loss')
            term_loss = -tf.reduce_mean(term_surrogate_loss, name='term_loss')

            total_loss = actor_loss
            if self.entropy_beta != 0:
                total_loss = actor_loss - entropy * self.entropy_beta
            if self.critic_beta != 0:
                total_loss = actor_loss + critic_loss * self.critic_beta
            total_loss += term_loss
            self.term_loss = term_loss
            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy

        return total_loss

    def _kl_entropy(self):
        # NOT FINISHED
        with tf.name_scope('kl_divergence'):
            target_logits = self.global_network.logits

            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
            a1 = target_logits - tf.reduce_max(target_logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, keepdims=True)
            z1 = tf.reduce_sum(ea1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)))

def build_network(input_hold):
    net = input_hold
    _layers = {'input': net}

    # Block 1 : Separable CNN
    net = layers.separable_conv2d(
            inputs=net,
            num_outputs=32,
            kernel_size=4,
            stride=2,
            padding='VALID',
            depth_multiplier=4,
        )
    _layers['sepCNN1'] = net

    # Block 2 : Attention (with residual connection)
    net, att_layers = non_local_nn_2d(net, 16, pool=False, name='non_local', return_layers=True)
    _layers['attention'] = att_layers['attention']
    _layers['NLNN'] = net

    # Block 3 : Convolution
    net = layers.convolution(inputs=net, num_outputs=64, kernel_size=3, stride=2, padding='VALID')
    _layers['CNN1'] = net
    net = layers.convolution(inputs=net, num_outputs=64, kernel_size=2, stride=2, padding='VALID')
    _layers['CNN2'] = net

    # Block 4 : Feature Vector
    net = layers.flatten(net)
    _layers['flat'] = net
    net = layers.fully_connected(
            net,
            128,
            activation_fn=None,
        )
    _layers['dense1'] = net

    return net, _layers

class V2_PPO_Probabilistic(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, critic_beta=0.5, name='PPO'):
        super(V2_PPO_Probabilistic, self).__init__(name=name)

        # Feature Encoder
        self.feature_network = V2()

        # Actor
        self.actor_dense1 = keras_layers.Dense(action_size)
        self.sftmx = keras_layers.Activation('softmax')

        # Critic
        self.critic_dense1 = keras_layers.Dense(1)


    def call(self, inputs):
        net = self.feature_network(inputs)

        logits = self.actor_dense1(net)
        logits = tf.math.maximum(logits, 1e-9)
        actor = self.sftmx(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic = self.critic_dense1(net)
        critic = tf.reshape(critic, [-1])

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.feature = net

        return actor, logits, log_logits, critic, net

    def build_loss(self, old_log_logit, action, advantage, td_target):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

            # Critic Loss
            td_error = td_target - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            total_loss = actor_loss
            if self.entropy_beta != 0:
                total_loss = actor_loss - entropy * self.entropy_beta
            if self.critic_beta != 0:
                total_loss = actor_loss + critic_loss * self.critic_beta

            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy

        return total_loss

    def _kl_entropy(self):
        # NOT FINISHED
        with tf.name_scope('kl_divergence'):
            target_logits = self.global_network.logits

            a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
            a1 = target_logits - tf.reduce_max(target_logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, keepdims=True)
            z1 = tf.reduce_sum(ea1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)))

def build_network(input_hold):
    net = input_hold
    _layers = {'input': net}

    # Block 1 : Separable CNN
    net = layers.separable_conv2d(
            inputs=net,
            num_outputs=32,
            kernel_size=4,
            stride=2,
            padding='VALID',
            depth_multiplier=4,
        )
    _layers['sepCNN1'] = net

    # Block 2 : Attention (with residual connection)
    net, att_layers = non_local_nn_2d(net, 16, pool=False, name='non_local', return_layers=True)
    _layers['attention'] = att_layers['attention']
    _layers['NLNN'] = net

    # Block 3 : Convolution
    net = layers.convolution(inputs=net, num_outputs=64, kernel_size=3, stride=2, padding='VALID')
    _layers['CNN1'] = net
    net = layers.convolution(inputs=net, num_outputs=64, kernel_size=2, stride=2, padding='VALID')
    _layers['CNN2'] = net

    # Block 4 : Feature Vector
    net = layers.flatten(net)
    _layers['flat'] = net
    net = layers.fully_connected(
            net,
            128,
            activation_fn=None,
        )
    _layers['dense1'] = net

    return net, _layers

if __name__=='__main__':
    network = V2()
    z = network(tf.placeholder(tf.float32, [None, 39, 39, 24]))

    network.summary()
    print(network.layers)
    print(network.trainable_variables)
