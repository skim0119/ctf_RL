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
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='elu'),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),])
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.Conv2D(filters=16, kernel_size=2, strides=2, activation='elu'),
            layers.MaxPool2D(),
            #Non_local_nn(4),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),
            layers.Dense(units=64, activation='elu'),])
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

class V4INV(tf.keras.Model):
    @store_args
    def __init__(self, trainable=True, name='FeatureNN_Inverse'):
        super(V4INV, self).__init__(name=name)

        # Feature Encoder
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')
        self.static_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=512, activation='elu'),
            layers.Reshape([4,4,32]),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, output_padding=1, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, output_padding=1, activation='tanh')])
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=1600, activation='elu'),
            layers.Reshape([10,10,16]),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2, activation='tanh')])

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

class V4Discentralized(tf.keras.Model):
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
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='elu'),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),])
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=dynamic_input_shape),
            layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='elu'),
            layers.MaxPool2D(),
            #Non_local_nn(4),
            layers.Flatten(),
            layers.Dense(units=64, activation='elu'),
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

class V4INVDiscentralized(tf.keras.Model):
    @store_args
    def __init__(self, trainable=True, name='FeatureNN_Inverse'):
        super(V4INV, self).__init__(name=name)

        # Feature Encoder
        self.dense1 = layers.Dense(units=V4.LATENT_DIM, activation='elu')
        self.static_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=2592, activation='elu'),
            layers.Reshape([9,9,32]),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, output_padding=1, activation='elu'),
            layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, output_padding=1, activation='tanh')])
        self.dynamic_network = keras.Sequential([
            layers.Input(shape=[V4.LATENT_DIM//2]),
            layers.Dense(units=5776, activation='elu'),
            layers.Reshape([19,19,16]),
            layers.UpSampling2D(),
            layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, output_padding=1, activation='tanh')])

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

    sample_size = 32
    image_shape = [40,40,6]
    sample_shape = [sample_size]+image_shape
    latent_size = 128
    latent_shape = [sample_size]+[latent_size]

    # Encoder Shape Summary
    model = V4(image_shape, 5)
    model.print_summary()

    sample = np.random.random(sample_shape).astype(np.float32)
    output = model(sample)
    print('input: ', sample.shape)
    print('output: ', output.shape)

    # Decoder Shape Summary
    model = V4INV()
    model.print_summary()

    sample = np.random.random(latent_shape).astype(np.float32)
    output = model(sample)
    print('input: ', sample.shape)
    print('output: ', output.shape)

