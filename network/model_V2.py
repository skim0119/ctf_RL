"""
Attention + sepCNN network for CtF encoder (June 11)

Mainly used for:
    A3C
    PPO
    VAE
"""

from functools import partial

import tensorflow as tf
#import tensorflow.contrib.layers as layers
import tensorflow.keras.layers as keras_layers

from network.attention import non_local_nn_2d
from network.attention import Non_local_nn
from utility.utils import store_args

from method.base import put_channels_on_grid

class V2(tf.keras.Model):
    def __init__(self, name='V2'):
        super(V2, self).__init__(name=name)
        self.sep_conv2d = keras_layers.SeparableConv2D(
                filters=32,
                kernel_size=4,
                strides=2,
                padding='valid',
                depth_multiplier=4,
                activation='relu',
            )
        self.non_local = Non_local_nn(32)
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

        return net 

def build_network2(input_hold):
    network = V2()
    return network(input_hold), network._layers_snapshot


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

