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

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, sess=None):
        if sess is None:
            sess = tf.Session(graph=tf.Graph())
        with sess.graph.as_default():
            super(CVAE, self).__init__()
            self.latent_dim = latent_dim
            self.inference_net = tf.keras.Sequential( [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                    ]
                )

            self.generative_net = tf.keras.Sequential( [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation='relu'),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation='relu'),
                    # No activation
                    tf.keras.layers.Conv2DTranspose(
                        filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
                    ]
                )

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits
