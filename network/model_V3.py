"""
Feature encoder with Attention and SepCNN (June 26)

Mainly used for:
    A3C
    PPO
    VAE
"""

import os
import sys
sys.path.append('/Users/skim0119/github/raide_rl')

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from network.attention import self_attention
from network.core import layer_normalization

from method.base import put_channels_on_grid

from utility.utils import store_args

class Spatial_VAE(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, input_placeholder, latent_dim=128, lr=1e-4, scope=None, reuse=tf.AUTO_REUSE):
        super().__init__()
        with tf.variable_scope(scope, 'vae', reuse=reuse) as scope:
            # Graph
            self.inference_net = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=input_shape, name='state_input'),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=5, strides=(3, 3)),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=3, strides=(2, 2)),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=2, strides=(1, 1)),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation("relu"),

                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                    ], name='encoder')

            self.generative_net = tf.keras.Sequential( [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,), name='latent_input'),
                    tf.keras.layers.Dense(units=4*4*64, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(4, 4, 64)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=3,
                        strides=(1, 1),
                        #padding="SAME",
                        activation='relu'),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=3,
                        strides=(2, 2),
                        #padding="SAME",
                        activation='relu'),
                    # No activation
                    tf.keras.layers.Conv2DTranspose(
                        filters=6, kernel_size=5, strides=(3, 3), padding="SAME", activation='tanh'),
                    ], name='decoder')

            with tf.name_scope('data_pipe'):
                self.mean, self.logvar = self.encode(input_placeholder)
                self.z = self.reparameterize(self.mean, self.logvar)

            # Sample
            with tf.name_scope('sample'):
                self.random_sample = self.sample()

            # Loss
            with tf.name_scope('loss'):
                self.x_logit = self.decode(self.z)
                #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logit, labels=input_placeholder)
                logpx_z = -tf.losses.mean_squared_error(predictions=self.x_logit, labels=input_placeholder)
                #logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
                logpz = self.log_normal_pdf(self.z, 0., 0.)
                logqz_x = self.log_normal_pdf(self.z, self.mean, self.logvar)
                self.elbo_loss = -tf.reduce_mean(logpx_z + 0.001*(logpz - logqz_x))

            # Gradient
            self.grads = tf.gradients(self.elbo_loss, self.trainable_variables)

            # Optimizer
            with tf.name_scope('trainer'):
                self.optimizer = tf.train.AdamOptimizer(lr)
                self.update = [self.optimizer.apply_gradients(zip(self.grads, self.trainable_variables))]

    def build_pipeline(self, input_placeholder, pipe_name='pipe'):
        with tf.name_scope(pipe_name):
            mean, logvar = self.encode(input_placeholder)
            z = self.reparameterize(mean, logvar)
        return z

    def build_external_loss(self, loss, pipe_name='extern_loss'):
        with tf.name_scope(pipe_name):
            grad = tf.gradients(self.elbo_loss, self.trainable_variables)
            update = self.optimizer.apply_gradient(zip(grad, self.trainable_variables))
            self.update.append(update)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=[4, self.latent_dim])
        return self.decode(eps, apply_tanh=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        num = tf.shape(mean)[0]
        eps = tf.random.normal(shape=[num, self.latent_dim])
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.tanh(logits)
            return probs

        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
              -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
              axis=raxis)

class Temporal_VAE(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, input_placeholder,
            latent_dim=64, lr=1e-4, scope=None, reuse=tf.AUTO_REUSE):
        super().__init__()
        with tf.variable_scope(scope, 'vae', reuse=reuse) as scope:
            # Graph
            self.inference_net = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=input_shape, name='state_input'),
                    tf.keras.layers.Conv1D(
                        filters=8, kernel_size=5, strides=3, activation='relu'),
                    tf.keras.layers.Conv1D(
                        filters=16, kernel_size=3, strides=2, activation='relu'),

                    # No activation
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(latent_dim + latent_dim),
                    ], name='encoder')

            self.generative_net = tf.keras.Sequential( [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,), name='latent_input'),
                    tf.keras.layers.Dense(units=336, activation=tf.nn.relu),
                    tf.keras.layers.Reshape(target_shape=(21, 1, 16)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=8,
                        kernel_size=(3, 1),
                        strides=(2, 1),
                        padding="SAME",
                        activation='relu'),
                    # No activation
                    tf.keras.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=(5, 1),
                        strides=(3, 1),
                        #padding="SAME"
                        ),
                    ], name='decoder')
            
            with tf.name_scope('data_pipe'):
                self.mean, self.logvar = self.encode(input_placeholder)
                self.z = self.reparameterize(self.mean, self.logvar)

            with tf.name_scope('train_pipe'):
                nbatch =tf.shape(input_placeholder)[0]
                spatial_dim =tf.shape(input_placeholder)[1]
                blind_last = tf.concat([input_placeholder[:,:,:-1], tf.zeros((nbatch, spatial_dim, 1))], axis=-1)
                future = input_placeholder[:,:,-1:]
                blinded_mean, blinded_logvar = self.encode(blind_last)
                blinded_z = self.reparameterize(blinded_mean, blinded_logvar)

            # Sample
            with tf.name_scope('sample'):
                self.random_sample = self.sample()

            # Loss
            with tf.name_scope('loss'):
                self.x_logit = self.decode(blinded_z)
                self.x_logit = tf.squeeze(self.x_logit, axis=-1)
                #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logit, labels=future)
                logpx_z = -tf.losses.mean_squared_error(predictions=self.x_logit, labels=future)
                #logpx_z = -tf.reduce_sum(mse , axis=[1, 2, 3])
                logpz = self.log_normal_pdf(blinded_z, 0., 0.)
                logqz_x = self.log_normal_pdf(blinded_z, blinded_mean, blinded_logvar)
                self.elbo_loss = -tf.reduce_mean(logpx_z + 0.001*(logpz - logqz_x))

            # Gradient
            self.grads = tf.gradients(self.elbo_loss, self.trainable_variables)

            # Optimizer
            with tf.name_scope('trainer'):
                self.optimizer = tf.train.AdamOptimizer(lr)
                self.update = [self.optimizer.apply_gradients(zip(self.grads, self.trainable_variables))]

    def build_pipeline(self, input_placeholder, pipe_name='pipe'):
        with tf.name_scope(pipe_name):
            mean, logvar = self.encode(input_placeholder)
            z = self.reparameterize(mean, logvar)
        return z

    def build_external_loss(self, loss, pipe_name='extern_loss'):
        with tf.name_scope(pipe_name):
            grad = tf.gradients(self.elbo_loss, self.trainable_variables)
            update = self.optimizer.apply_gradient(zip(grad, self.trainable_variables))
            self.update.append(update)

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=[100, self.latent_dim])
        return self.decode(eps, apply_tanh=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        num = tf.shape(mean)[0]
        eps = tf.random.normal(shape=[num, self.latent_dim])
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.tanh(logits)
            return probs

        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
              -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
              axis=raxis)

def build_network(input_hold, output_size=128, return_layers=False):
    keep_dim = 4
    spatial_encoded = []

    frames = tf.split(input_hold, num_or_size_splits=keep_dim, axis=3)
    #vae_train_pipe = tf.placeholder(tf.float32, shape=[None,39,39,6])
    vae = Spatial_VAE((39,39,6), frames[-1], scope='vae', lr=1e-4)

    for frame in frames[:-1]:
        z = vae.build_pipeline(frame)
        spatial_encoded.append(z)
    spatial_encoded.append(vae.z)

    spatial_matrix = tf.stack(spatial_encoded, axis=-1)  # (None, 128, 4)
    spatial_matrix = tf.stop_gradient(spatial_matrix)
    tvae = Temporal_VAE((128, 4), spatial_matrix, scope='tvae', lr=1e-4)

    feature = tf.concat([z, tvae.z], axis=1) # (None, 192)

    #attention = self_attention(feature, 128, output_size)

    train_ops = [vae.update, tvae.update]
    #train_ops = [vae.update]

    return feature, train_ops

if __name__ == '__main__':
    #data = np.random.sample((100,2))
    #iter = dataset.make_initializable_iterator() # create the iterator
    #el = iter.get_next()
    #with tf.Session() as sess:
    #    # feed the placeholder with data
    #    sess.run(iter.initializer, feed_dict={ x: data }) 
    #    print(sess.run(el)) # output [ 0.52374458  0.71968478]

    keep_dim = 4
    spatial_encoded = []

    vae_train_pipe = tf.placeholder(tf.float32, shape=[None,39,39,6])
    vae = Spatial_VAE((39,39,6), vae_train_pipe, scope='vae')

    input_ph = tf.placeholder(tf.float32, [None,39,39,6*keep_dim])
    for frame in tf.split(input_ph, num_or_size_splits=keep_dim, axis=3):
        z = vae.build_pipeline(frame)
        spatial_encoded.append(z)


    #for layer in vae.inference_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)
    #for layer in vae.generative_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)

    #for variable in vae.trainable_variables:
    #    print(variable)

    spatial_matrix = tf.stack(spatial_encoded, axis=-1)  # (None, 128, 4)
    tvae = Temporal_VAE((128, 4), spatial_matrix, scope='tvae')
    #for layer in tvae.inference_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)
    #for layer in tvae.generative_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)

    feature = tf.concat([z, tvae.z], axis=1) # (None, 192)

    attention = self_attention(feature, 128, 128)

    checkpoint_directory = "./tmp/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    checkpoint = tf.train.Checkpoint()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint.save(file_prefix=checkpoint_prefix, session=sess)
        I = np.random.random(size=(2,39,39,24))
        feed_dict = {input_ph:I}
        update_feed_dict = {vae_train_pipe:np.random.random(size=(2,39,39,6))}
        sess.run(vae.update, update_feed_dict)
        sess.run(vae.update, update_feed_dict)
        sess.run(vae.update, update_feed_dict)
        file_writer = tf.summary.FileWriter(logdir='./tmp/logs/test', graph=sess.graph)


    print('graph build done') 
