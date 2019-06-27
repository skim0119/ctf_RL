"""
Feature encoder with Attention and SepCNN (June 26)

Mainly used for:
    A3C
    PPO
    VAE
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

from network.attention import non_local_nn_2d
from network.core import layer_normalization

from method.base import put_channels_on_grid

def build_network(input_hold, output_size=128, return_layers=False):
    net = input_hold
    _layers = {'input': net}

    # Block 1 : Separable CNN
    net = layers.separable_conv2d(
            inputs=net,
            num_outputs=64,
            kernel_size=5,
            stride=2,
            padding='VALID',
            depth_multiplier=4,
        )
    _layers['sepCNN1'] = net

    # Block 2 : Convolution
    net = layers.convolution(inputs=net, num_outputs=128, kernel_size=3, stride=1, padding='VALID')
    _layers['CNN1'] = net
    net = layers.max_pool2d(net, 2)
    net = layers.convolution(inputs=net, num_outputs=128, kernel_size=2, stride=1, padding='VALID')
    _layers['CNN2'] = net
    net = layers.max_pool2d(net, 2)

    # Block 3 : Attention (with residual connection)
    net, att_layers = non_local_nn_2d(net, 32, pool=False, use_dense=True, normalize=True, name='non_local', return_layers=True)
    _layers['attention'] = att_layers['attention']
    _layers['NLNN'] = net

    # Block 4 : Feature Vector
    net = tf.reduce_max(net, axis=-2) # Feature-wise max pool
    net = layers.flatten(net)
    _layers['flat'] = net
    net = layers.fully_connected(
            net,
            output_size,
            activation_fn=None,
        )
    _layers['dense1'] = net


    if return_layers:
        return net, _layers
    else:
        return net
