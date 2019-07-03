"""
Attention + sepCNN network for CtF encoder (June 11)

Mainly used for:
    A3C
    PPO
    VAE
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

from network.attention import non_local_nn_2d

from method.base import put_channels_on_grid

def build_network(input_hold, output_size=128, return_layers=False):
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
            output_size,
            activation_fn=None,
        )
    _layers['dense1'] = net


    if return_layers:
        return net, _layers
    else:
        return net

