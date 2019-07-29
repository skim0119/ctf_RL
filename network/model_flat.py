"""
Feature encoder with Attention and SepCNN (June 26)

Mainly used for:
    A3C
    PPO
    VAE
"""

import os
import sys
sys.path.append('/home/skim449/github/raide_rl')
sys.path.append('/Users/skim0119/github/raide_rl')
sys.path.append('/Users/namsong/github/raide_rl')

import tensorflow as tf
import tensorflow.keras.layers as keras_layers

import numpy as np

from network.attention import self_attention
from network.attention import Non_local_nn
from network.core import layer_normalization

from method.base import put_channels_on_grid

from utility.utils import store_args

class Flat_NN(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, input_placeholder, latent_dim=128, lr=1e-4, num_stack=4, scope=None):
        super().__init__()
        with tf.variable_scope(scope, 'nn') as scope:
            # Graph
            self.net = tf.keras.Sequential([
                    keras_layers.InputLayer(input_shape=input_shape, name='state_input'),
                    keras_layers.Flatten(),
                    keras_layers.Dense(2048, activation='elu'),
                    keras_layers.Dense(512, activation='elu'),
                    # No activation
                    keras_layers.Dense(latent_dim),
                    ], name='encoder')
            
            # Build data frame pipe (keep last)
            self.z_list = []
            frames = tf.split(input_placeholder, num_or_size_splits=num_stack, axis=3)
            for frame in frames:
                z = self.build_pipeline(frame, pipe_name='frame_pipe')
                self.z_list.append(z)
            self.z = z

    def build_pipeline(self, input_ph, pipe_name='pipe'):
        with tf.name_scope(pipe_name):
            z = self.net(input_ph)
        return z

def build_network(input_hold, output_size=128, return_layers=False, keep_dim=4):
    svae = Flat_NN((39,39,6), input_hold, scope='spatial')

    feature = svae.z

    encoding_var = svae.trainable_variables 

    return feature, encoding_var

if __name__ == '__main__':
    input_hold = tf.placeholder(dtype=tf.float32, shape=[None, 39, 39, 24])
    svae = Flat_NN((39,39,6), input_hold, scope='spatial_vae', lr=1e-4, num_stack=4)

    for layer in svae.inference_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)
    for layer in svae.generative_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)

    for variable in svae.trainable_variables:
        print(variable)


    print('graph build done') 
