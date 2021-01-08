if __name__=='__main__':
    import sys
    sys.path.append("../")
    sys.path.append("./")

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from network.attention import Non_local_nn
from utility.utils import store_args

import numpy as np

# Model contains feature encoding architecture


class CentralEnc(tf.keras.Model):
    @store_args
    def __init__(self, input_shape,
                 trainable=True, name='CentralEncoder'):
        super(CentralEnc, self).__init__(name=name)

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=128, activation='relu'),
            #layers.GRU(64),
            layers.Dense(units=128, activation='relu'),
        ])

    def print_summary(self):
        self.static_network.summary()

    def call(self, inputs):
        net = self.static_network(inputs)
        return net

class DecentralEnc(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, action_space,
                 trainable=True, name='DecentralEnc'):
        super(DecentralEnc, self).__init__(name=name)

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=128, activation='relu'),
            #layers.GRU(64),
            layers.Dense(units=128, activation='relu'),
        ])

    def print_summary(self):
        self.static_network.summary()

    def call(self, inputs):
        net = self.static_network(inputs)

        return net

class DecentralDec(tf.keras.Model):
    @store_args
    def __init__(self, output_shape, trainable=True, name='DecentralDec'):
        super(DecentralDec, self).__init__(name=name)

        # Feature Encoder
        self.dense1 = layers.Dense(units=128, activation='elu')
        self.dense2 = layers.Dense(units=128, activation='elu')
        self.dense3 = layers.Dense(units=output_shape, activation='linear')

    def call(self, inputs):
        net = self.dense1(inputs)
        net = self.dense2(net)
        net = self.dense3(net)
        return net

