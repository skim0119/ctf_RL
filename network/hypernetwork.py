import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import numpy as np

class Hypernetwork1(tf.keras.Model):
    def __init__(self, input_shape, dim=64):
        super(Hypernetwork1, self).__init__()
        self.input_dim = input_shape[-1]
        self.dim = dim

        embedded_dim = self.input_dim * dim
        self.hw1 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=embedded_dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, self.input_dim, dim]),
        ])
        self.hw2 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim*dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, dim, dim])
        ])
        self.hb1 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, 1, dim])
        ])
        self.hb2 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, 1, dim])
        ])
        self.flatten = layers.Flatten()
        
    def call(self, inputs):
        batch_size = inputs.shape[0]
        w1 = tf.math.abs(self.hw1(inputs)) # [-1, 4, 80, 64]
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
        self.input_dim = input_shape[-1]
        self.dim = dim

        embedded_dim = self.input_dim * dim
        self.hw1 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=embedded_dim, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=embedded_dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, self.input_dim, dim]),
        ])
        self.hw2 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim*dim, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=dim*dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, dim, dim])
        ])
        self.hb1 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, 1, dim])
        ])
        self.hb2 = keras.Sequential([
            layers.Input(shape=self.input_dim),
            layers.Dense(units=dim, activation='elu',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Dense(units=dim, activation='linear',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
            layers.Reshape([-1, 1, dim])
        ])
        self.flatten = layers.Flatten()
        
    def call(self, inputs):
        batch_size = inputs.shape[0]
        w1 = tf.math.abs(self.hw1(inputs)) # [-1, 4, 80, 64]
        b1 = self.hb1(inputs) # [-1, 4, 1, 64]
        inputs = tf.expand_dims(inputs, axis=1)  # [-1, 4, 1, 80]
        hidden = tf.nn.elu(tf.matmul(inputs, w1) + b1)  # [-1, 4, 1, 64]

        w2 = self.hw2(inputs) # [-1, 4, 64, 64]
        b2 = self.hb2(inputs) # [-1, 4, 1, 64]
        hidden = tf.nn.elu(tf.matmul(hidden, w2)+b2) # [-1, 4, 1, 64]

        hidden = self.flatten(hidden) # [-1, 4*64]
        return hidden

class Hypernetwork2c(tf.keras.Model):
    def __init__(self, input_shape, state_shape, dim=64):
        super(Hypernetwork2c, self).__init__()

        embedded_dim = 32
        self.hw1 = keras.Sequential([
            layers.Input(shape=state_shape),
            layers.Dense(units=dim, activation='relu'),
            layers.Dense(units=input_shape[0]*embedded_dim, activation='linear'),
            layers.Reshape([input_shape[0], embedded_dim]),
        ])
        self.hw2 = keras.Sequential([
            layers.Input(shape=state_shape),
            layers.Dense(units=dim, activation='relu'),
            layers.Dense(units=embedded_dim, activation='linear'),
            layers.Reshape([embedded_dim, 1])
        ])
        self.hb1 = keras.Sequential([
            layers.Input(shape=state_shape),
            layers.Dense(units=embedded_dim, activation='linear'),
            layers.Reshape([1, embedded_dim])
        ])
        self.hb2 = keras.Sequential([
            layers.Input(shape=state_shape),
            layers.Dense(units=embedded_dim, activation='relu'),
            layers.Dense(units=1, activation='linear'),
            layers.Reshape([1, 1])
        ])
        
    def call(self, inputs, state):
        # inputs: [-1, 20]
        # state: [-1, 80]
        batch_size = state.shape[0]
        inputs = tf.expand_dims(inputs, axis=1)  # [-1, 1, 20]
        w1 = tf.math.abs(self.hw1(state)) # [-1, 20, 64]
        b1 = self.hb1(state) # [-1, 1, 64]
        hidden = tf.nn.relu(tf.matmul(inputs, w1) + b1)  # [-1, 1, 64]

        w2 = self.hw2(state) # [-1, 64, 1]
        b2 = self.hb2(state) # [-1, 1, 1]
        hidden = tf.matmul(hidden, w2)+b2 # [-1, 1, 1]

        hidden = tf.reshape(hidden, [-1])

        return hidden

