"""
a3c.py module includes basic modules and implementation of A3C for CtF environment.

Some of the methods are left unimplemented which makes the a3c module to be parent abstraction.

Script contains example A3C

TODO:
    - Include gradient and weight histograph for nan debug
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from method.base import Deep_layer
from method.pg import Loss, Backpropagation

from method.base import Tensorboard_utility as TB
from method.base import put_channels_on_grid
from method.a3c import a3c

from network.attention_ctf import build_network

class A3C_attention(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(A3C_attention, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def run_network(self, states):
        actions = []
        critics = []
        for idx, state in enumerate(states):
            feed_dict = {
                self.state_input: state[np.newaxis,:]
            }
            ops = [self.actor, self.critic]
            prob, critic = self.sess.run(ops, feed_dict)
            action = np.random.choice(self.action_size, p=prob[0] / sum(prob[0])) 
            actions.append(action)
            critics.append(critic[0])
        return actions, critics

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    stride=2,
                    padding='VALID',
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    stride=2,
                    padding='VALID',
                    depth_multiplier=1,
                )
            self.feature_static = net_static
            self.feature_dynamic = net_dynamic
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)
            add_image(net, 'attention')
            self.feature_attention = net

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3, stride=2, padding='VALID')
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2, stride=2, padding='VALID')
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = input_hold

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=5,
                    stride=2,
                    padding='VALID',
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=5,
                    stride=2,
                    padding='VALID',
                    depth_multiplier=1,
                )
            net = tf.concat([net_static, net_dynamic], axis=-1)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3, stride=2, padding='VALID')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2, stride=1, padding='VALID')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
            
        return logits, actor, critic, a_vars, c_vars

class V1_old(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(V1_old, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            self.feature_static = net_static
            self.feature_dynamic = net_dynamic
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            self.feature_attention = net
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[32, 64, 64],
                kernels=[5, 3, 2],
                pools=[2, 2, 2],
                flatten=True
            )

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
            
        return logits, actor, critic, a_vars, c_vars

class V1_old_attention(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(V1_old_attention, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            self.feature_static = net_static
            self.feature_dynamic = net_dynamic
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)
            add_image(net, 'attention')
            self.feature_attention = net

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = Deep_layer.conv2d_pool(
                input_layer=input_hold,
                channels=[32, 64, 64],
                kernels=[5, 3, 2],
                pools=[2, 2, 2],
                flatten=True
            )

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
            
        return logits, actor, critic, a_vars, c_vars

class self_att_v1_roomba(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(self_att_v1_roomba, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def run_network(self, states):
        actions = []
        critics = []
        for idx, state in enumerate(states):
            feed_dict = {
                self.state_input: state[np.newaxis,:]
            }
            ops = [self.actor, self.critic]
            prob, critic = self.sess.run(ops, feed_dict)
            action = np.random.choice(self.action_size, p=prob[0] / sum(prob[0])) 
            actions.append(action)
            critics.append(critic[0])
        return actions, critics

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            self.feature_static = net_static
            self.feature_dynamic = net_dynamic
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)
            add_image(net, 'attention')
            self.feature_attention = net

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = input_hold

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            net = tf.concat([net_static, net_dynamic], axis=-1)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
            
        return logits, actor, critic, a_vars, c_vars

class v1_roomba_noatt(a3c):
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module provides simplest template for using a3c module prescribed above.

    """

    def __init__(self, in_size, action_size, scope,
                 lr_actor=1e-4, lr_critic=1e-4,
                 entropy_beta=0.01,
                 sess=None, global_network=None,
                 **kwargs):
        """ Initialize AC network and required parameters """
        super(v1_roomba_noatt, self).__init__(
            in_size, action_size, scope,
            lr_actor, lr_critic,
            entropy_beta, sess, global_network,
            **kwargs)

    def run_network(self, states):
        actions = []
        critics = []
        for idx, state in enumerate(states):
            feed_dict = {
                self.state_input: state[np.newaxis,:]
            }
            ops = [self.actor, self.critic]
            prob, critic = self.sess.run(ops, feed_dict)
            action = np.random.choice(self.action_size, p=prob[0] / sum(prob[0])) 
            actions.append(action)
            critics.append(critic[0])
        return actions, critics

    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            self.feature_static = net_static
            self.feature_dynamic = net_dynamic
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            net = tf.contrib.layers.max_pool2d(net, 2)

            self.feature_attention = net

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = input_hold

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            net = tf.concat([net_static, net_dynamic], axis=-1)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
            
        return logits, actor, critic, a_vars, c_vars