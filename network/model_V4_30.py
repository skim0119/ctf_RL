if __name__=='__main__':
    import sys
    sys.path.append("../")
    sys.path.append("./")

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

from network.attention import Non_local_nn
from network.lstm_utils import LSTM_Flatten
from utility.utils import store_args

import numpy as np

# Model contains feature encoding architecture


class V4(tf.keras.Model):
    STATIC_CHANNEL = [0,1,3]
    DYNAMIC_CHANNEL = [2,4,5]
    LATENT_DIM = 128

    @store_args
    def __init__(self, input_shape, action_size=5,
                 trainable=True, name='FeatureNN'):
        super(V4, self).__init__(name=name)

        static_input_shape = [input_shape[0], input_shape[1], len(V4.STATIC_CHANNEL)]
        dynamic_input_shape = [input_shape[0], input_shape[1], len(V4.DYNAMIC_CHANNEL)]

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=static_input_shape),
            layers.SeparableConv2D(
                filters=16, kernel_size=4, strides=2,
                padding='valid', depth_multiplier=8, activation='elu'),
            layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='elu'),
            layers.MaxPool2D(),
            layers.Conv2D(filters=32, kernel_size=2, strides=1, activation='elu'),
            layers.Flatten(),
            layers.Dense(units=128, activation='elu'),], name='static_network')
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='elu'),
            layers.MaxPool2D(),
            layers.Conv2D(filters=16, kernel_size=2, strides=1, activation='elu'),
            layers.Flatten(),
            layers.Dense(units=128, activation='elu'),], name='dynamic_network')
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        static = tf.gather(inputs, V4.STATIC_CHANNEL, axis=-1)
        dynamic = tf.gather(inputs, V4.DYNAMIC_CHANNEL, axis=-1)

        static_net = self.static_network(static)
        dynamic_net = self.dynamic_network(dynamic)
        net = tf.concat([static_net, dynamic_net], axis=-1)

        net = self.dense1(net)

        return net

class V4_lstm(tf.keras.Model):
    STATIC_CHANNEL = [0,1,3]
    DYNAMIC_CHANNEL = [2,4,5]
    LATENT_DIM = 128

    @store_args
    def __init__(self, input_shape, action_size=5,
                 trainable=True, name='FeatureNN'):
        super(V4_lstm, self).__init__(name=name)

        static_input_shape = [input_shape[0], input_shape[1],input_shape[2], len(V4.STATIC_CHANNEL)]
        dynamic_input_shape = [input_shape[0], input_shape[1],input_shape[2], len(V4.DYNAMIC_CHANNEL)]
        print(static_input_shape)
        print(dynamic_input_shape)

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=static_input_shape),
            layers.TimeDistributed(layers.SeparableConv2D(
                filters=16, kernel_size=4, strides=2,
                padding='valid', depth_multiplier=8, activation='elu')),
            layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='elu')),
            layers.TimeDistributed(layers.MaxPool2D()),
            layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=2, strides=1, activation='elu')),
            layers.TimeDistributed(layers.Flatten()),
            layers.TimeDistributed(layers.Dense(units=128, activation='elu')),
            layers.LSTM(128),
            ], name='static_network')
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='elu')),
            layers.TimeDistributed(layers.MaxPool2D()),
            layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=2, strides=1, activation='elu')),
            layers.TimeDistributed(layers.Flatten()),
            layers.TimeDistributed(layers.Dense(units=128, activation='elu')),
            layers.LSTM(128),
            ], name='dynamic_network')
        self.dense1 = layers.Dense(units=V4_lstm.LATENT_DIM, activation='elu')

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        static = tf.gather(inputs, V4_lstm.STATIC_CHANNEL, axis=-1)
        dynamic = tf.gather(inputs, V4_lstm.DYNAMIC_CHANNEL, axis=-1)

        static_net = self.static_network(static)
        dynamic_net = self.dynamic_network(dynamic)
        net = tf.concat([static_net, dynamic_net], axis=-1)

        net = self.dense1(net)

        return net

class V4INV(tf.keras.Model):
    @store_args
    def __init__(self, trainable=True, name='FeatureNN_Inverse'):
        super(V4INV, self).__init__(name=name)

        # Feature Encoder
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')
        self.static_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=576, activation='elu'),
            layers.Reshape([6,6,16]),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2, activation='linear')],
            name='static_network')
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=576, activation='elu'),
            layers.Reshape([6,6,16]),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='elu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2, activation='linear')],
            name='dynamic_network')

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        net = self.dense1(inputs)
        static, dynamic = tf.split(net, 2, axis=-1)

        static = self.static_network(static)
        dynamic = self.dynamic_network(dynamic)
        net = tf.concat([static, dynamic], axis=-1)
        net = tf.gather(net, [0,1,3,2,4,5], axis=-1)

        return net

class V4Decentral(tf.keras.Model):
    STATIC_CHANNEL = [0,1,3]
    DYNAMIC_CHANNEL = [2,4,5]
    LATENT_DIM = 128

    @store_args
    def __init__(self, input_shape, action_size=5,
                 trainable=True, name='FeatureNN'):
        super(V4Decentral, self).__init__(name=name)

        static_input_shape = [input_shape[0], input_shape[1], len(V4.STATIC_CHANNEL)]
        dynamic_input_shape = [input_shape[0], input_shape[1], len(V4.DYNAMIC_CHANNEL)]

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=static_input_shape),
            layers.SeparableConv2D(
                filters=16, kernel_size=4, strides=2,
                padding='valid', depth_multiplier=8, activation='elu'),
            layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='elu'),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),])
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='elu'),
            layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='elu'),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),])
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')
        #self.dout1 = layers.Dropout(rate=0.2)

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        static = tf.gather(inputs, V4.STATIC_CHANNEL, axis=-1)
        dynamic = tf.gather(inputs, V4.DYNAMIC_CHANNEL, axis=-1)

        static_net = self.static_network(static)
        dynamic_net = self.dynamic_network(dynamic)
        net = tf.concat([static_net, dynamic_net], axis=-1)

        net = self.dense1(net)
        #net = self.dout1(net)

        return net

class V4Decentral_lstm(tf.keras.Model):
    STATIC_CHANNEL = [0,1,3]
    DYNAMIC_CHANNEL = [2,4,5]
    LATENT_DIM = 128

    @store_args
    def __init__(self, input_shape, action_size=5,
                 trainable=True, name='FeatureNN'):
        super(V4Decentral_lstm, self).__init__(name=name)

        static_input_shape = [input_shape[0], input_shape[1],input_shape[2], len(V4.STATIC_CHANNEL)]
        dynamic_input_shape = [input_shape[0], input_shape[1],input_shape[2], len(V4.DYNAMIC_CHANNEL)]

        # Feature Encoder
        self.static_network = keras.Sequential([
            layers.Input(shape=static_input_shape),
            layers.TimeDistributed(layers.SeparableConv2D(
                filters=16, kernel_size=4, strides=2,
                padding='valid', depth_multiplier=8, activation='elu')),
            layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='elu')),
            layers.TimeDistributed(layers.Flatten()),
            layers.TimeDistributed(layers.Dense(units=128, activation='elu')),
            layers.LSTM(128),
            ], name='static_network')
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='elu')),
            layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='elu')),
            layers.TimeDistributed(layers.Flatten()),
            layers.TimeDistributed(layers.Dense(units=128, activation='elu')),
            layers.LSTM(128),
            ], name='dynamic_network')
        self.dense1 = layers.Dense(units=V4Decentral_lstm.LATENT_DIM, activation='elu')

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        static = tf.gather(inputs, V4Decentral_lstm.STATIC_CHANNEL, axis=-1)
        dynamic = tf.gather(inputs, V4Decentral_lstm.DYNAMIC_CHANNEL, axis=-1)

        static_net = self.static_network(static)
        dynamic_net = self.dynamic_network(dynamic)
        net = tf.concat([static_net, dynamic_net], axis=-1)

        net = self.dense1(net)

        return net

class V4INVDecentral(tf.keras.Model):
    @store_args
    def __init__(self, trainable=True, name='FeatureNN_Inverse', **kwargs):
        super(V4INVDecentral, self).__init__()

        # Feature Encoder
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')
        self.static_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=3136, activation='elu'),
            layers.Reshape([14,14,16]),
            #layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, output_padding=0, activation='elu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, output_padding=0, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation='linear')],
            name='static_network')
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=3136, activation='elu'),
            layers.Reshape([14,14,16]),
            #layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, output_padding=0, activation='elu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, output_padding=0, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, activation='linear')],
            name='dynamic_network')

    def print_summary(self):
        self.static_network.summary()
        self.dynamic_network.summary()

    def call(self, inputs):
        net = self.dense1(inputs)
        static, dynamic = tf.split(net, 2, axis=-1)

        static = self.static_network(static)
        dynamic = self.dynamic_network(dynamic)
        net = tf.concat([static, dynamic], axis=-1)
        net = tf.gather(net, [0,1,3,2,4,5], axis=-1)

        return net

if __name__=='__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print('centralized network:')
    sample_size = 32
    map_size = 30
    map_size = map_size*2-1
    image_shape = [map_size,map_size,6]
    sample_shape = [sample_size]+image_shape
    latent_size = 128
    latent_shape = [sample_size]+[latent_size]

    # Encoder Shape Summary
    model = V4(image_shape, 5)
    model.print_summary()

    sample = np.random.random(sample_shape).astype(np.float32)
    output = model(sample)
    print('c-inc-input: ', sample.shape)
    print('c-inc-output: ', output.shape)

    # Decoder Shape Summary
    model = V4INV()
    model.print_summary()

    sample = np.random.random(latent_shape).astype(np.float32)
    output = model(sample)
    print('c-dec-input: ', sample.shape)
    print('c-dec-output: ', output.shape)

    # LSTM Shape Summary
    image_shape = [3,map_size,map_size,6]
    sample_shape = [sample_size]+image_shape

    model = V4_lstm(image_shape, 5)
    model.print_summary()

    sample = np.random.random(sample_shape).astype(np.float32)
    output = model(sample)
    print('c-dec-lstm-input: ', sample.shape)
    print('c-dec-lstm-output: ', output.shape)


    print('decentralized network:')
    sample_size = 32
    #map_size = map_size*2-1
    image_shape = [map_size,map_size,6]
    sample_shape = [sample_size]+image_shape
    latent_size = 128
    latent_shape = [sample_size]+[latent_size]

    # Encoder Shape Summary
    model = V4Decentral(image_shape, 5)
    model.print_summary()

    sample = np.random.random(sample_shape).astype(np.float32)
    output = model(sample)
    print('d-inc-input: ', sample.shape)
    print('d-inc-output: ', output.shape)

    # Decoder Shape Summary
    model = V4INVDecentral()
    model.print_summary()

    sample = np.random.random(latent_shape).astype(np.float32)
    output = model(sample)
    print('d-dec-input: ', sample.shape)
    print('d-dec-output: ', output.shape)

    # LSTM Shape Summary
    image_shape = [3,map_size,map_size,6]
    sample_shape = [sample_size]+image_shape
    model = V4Decentral_lstm(image_shape)
    model.print_summary()

    sample = np.random.random(sample_shape).astype(np.float32)
    output = model(sample)
    print('d-dec-lstm-input: ', sample.shape)
    print('d-dec-lstm-output: ', output.shape)
