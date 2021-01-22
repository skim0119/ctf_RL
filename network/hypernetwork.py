import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np

class Hypernetwork1(tf.keras.Model):
    def __init__(self, input_shape, dim=64):
        super(Hypernetwork2, self).__init__()
        self.keepframe = input_shape[0]
        self.input_dim = input_shape[-1]
        self.dim = dim

        embedded_dim = self.input_dim * dim
        self.hw1 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=embedded_dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, self.input_dim, dim]),
        ])
        self.hw2 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim*dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, dim, dim])
        ])
        self.hb1 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, 1, dim])
        ])
        self.hb2 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, 1, dim])
        ])
        self.flatten = layers.Reshape([-1, self.keepframe * dim])
        
    def call(self, inputs):
        batch_size = inputs.shape[0]
        w1 = self.hw1(inputs) # [-1, 4, 80, 64]
        b1 = self.hb1(inputs) # [-1, 4, 1, 64]
        inputs = tf.expand_dims(inputs, axis=1)  # [-1, 4, 1, 80]
        hidden = tf.nn.elu(tf.matmul(inputs, w1) + b1)  # [-1, 4, 1, 64]

        w2 = self.hw2(inputs) # [-1, 4, 64, 64]
        b2 = self.hb2(inputs) # [-1, 4, 1, 64]
        hidden = tf.nn.elu(tf.matmul(hidden, w2)+b2) # [-1, 4, 1, 64]

        hidden = self.flatten(hidden) # [-1, 4*64]
        return hidden

class Hypernetwork2(tf.keras.Model):
    def __init__(self, input_shape, dim=64):
        super(Hypernetwork2, self).__init__()
        self.keepframe = input_shape[0]
        self.input_dim = input_shape[-1]
        self.dim = dim

        embedded_dim = self.input_dim * dim
        self.hw1 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=embedded_dim//4, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=embedded_dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, self.input_dim, dim]),
        ])
        self.hw2 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim*dim//4, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=dim*dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, dim, dim])
        ])
        self.hb1 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, 1, dim])
        ])
        self.hb2 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units=dim//4, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([self.keepframe, 1, dim])
        ])
        self.flatten = layers.Reshape([-1, self.keepframe * dim])
        
    def call(self, inputs):
        batch_size = inputs.shape[0]
        w1 = self.hw1(inputs) # [-1, 4, 80, 64]
        b1 = self.hb1(inputs) # [-1, 4, 1, 64]
        inputs = tf.expand_dims(inputs, axis=1)  # [-1, 4, 1, 80]
        hidden = tf.nn.elu(tf.matmul(inputs, w1) + b1)  # [-1, 4, 1, 64]

        w2 = self.hw2(inputs) # [-1, 4, 64, 64]
        b2 = self.hb2(inputs) # [-1, 4, 1, 64]
        hidden = tf.nn.elu(tf.matmul(hidden, w2)+b2) # [-1, 4, 1, 64]

        hidden = self.flatten(hidden) # [-1, 4*64]
        return hidden

