"""
Feature encoder with Attention and SepCNN (June 26)

Mainly used for:
    A3C
    PPO
    VAE
"""

import os
import sys
sys.path.append('/home/skim449/github/raide_rl')
sys.path.append('/Users/namsong/github/raide_rl')

import tensorflow as tf
import tensorflow.keras.layers as keras_layers

import numpy as np

from network.attention import self_attention
from network.core import layer_normalization

from method.base import put_channels_on_grid

from utility.utils import store_args

class Spatial_VAE(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, input_placeholder, latent_dim=128, lr=1e-4, num_stack=4, scope=None):
        super().__init__()
        with tf.variable_scope(scope, 'vae') as scope:
            # Graph
            self.inference_net = tf.keras.Sequential([
                    keras_layers.InputLayer(input_shape=input_shape, name='state_input'),
                    keras_layers.Conv2D(filters=32, kernel_size=5, activation='elu'),
                    #keras_layers.BatchNormalization(),
                    keras_layers.AvgPool2D((3,3)),
                    #keras_layers.Conv2D(filters=64, kernel_size=3),
                    #keras_layers.BatchNormalization(),
                    #keras_layers.Activation("elu"),
                    #keras_layers.AvgPool2D((2,2)),
                    keras_layers.Conv2D(filters=64, kernel_size=3, activation='elu'),
                    keras_layers.AvgPool2D((2,2)),
                    #keras_layers.BatchNormalization(),

                    keras_layers.Flatten(),
                    # No activation
                    keras_layers.Dense(latent_dim + latent_dim),
                    ], name='encoder')
            
            self.generative_net = tf.keras.Sequential( [
                    keras_layers.InputLayer(input_shape=(latent_dim,), name='latent_input'),
                    keras_layers.Dense(units=4*4*64, activation=tf.nn.elu),
                    keras_layers.Reshape(target_shape=(4, 4, 64)),
                    keras_layers.UpSampling2D((2,2)),
                    keras_layers.ZeroPadding2D((1,1)),
                    keras_layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=3,
                        #padding="SAME",
                        activation='elu'),
                    keras_layers.UpSampling2D((3,3)),
                    # No activation
                    keras_layers.Conv2DTranspose(
                        filters=6,
                        kernel_size=5,
                        #padding="SAME",
                        activation='tanh',
                        ),
                    keras_layers.Cropping2D(cropping=((1,0),(1,0))),
                    ], name='decoder')

            # Build data frame pipe (keep last)
            self.z_list = []
            frames = tf.split(input_placeholder, num_or_size_splits=num_stack, axis=3)
            for frame in frames:
                z, mean, logvar = self.build_pipeline(frame, pipe_name='frame_pipe')
                self.z_list.append(z)
            self.z = z
            self.mean = mean
            self.logvar = logvar

            # Sample
            with tf.name_scope('sample'):
                self.random_sample = self.sample()

            # Loss
            with tf.name_scope('loss'):
                self.x_logit = self.decode(self.z)
                #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logit, labels=input_placeholder)
                logpx_z = -tf.losses.mean_squared_error(predictions=self.x_logit, labels=frames[-1])
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
        return z, mean, logvar

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
        # log normal probability distribution function
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
              -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
              axis=raxis)

class Temporal_VAE(tf.keras.Model):
    @store_args
    def __init__(self, input_shape, input_placeholder,
            latent_dim=64, lr=1e-4, scope=None):
        super().__init__()
        with tf.variable_scope(scope, 'vae') as scope:
            # Graph
            self.inference_net = tf.keras.Sequential([
                    keras_layers.InputLayer(input_shape=input_shape, name='state_input'),
                    keras_layers.Conv1D(
                        filters=8, kernel_size=5, strides=3, activation='elu'),
                    keras_layers.Conv1D(
                        filters=16, kernel_size=3, strides=2, activation='elu'),

                    # No activation
                    keras_layers.Flatten(),
                    keras_layers.Dense(latent_dim + latent_dim),
                    ], name='encoder')

            self.generative_net = tf.keras.Sequential( [
                    keras_layers.InputLayer(input_shape=(latent_dim,), name='latent_input'),
                    keras_layers.Dense(units=336, activation=tf.nn.elu),
                    keras_layers.Reshape(target_shape=(21, 1, 16)),
                    keras_layers.Conv2DTranspose(
                        filters=8,
                        kernel_size=(3, 1),
                        strides=(2, 1),
                        padding="SAME",
                        activation='elu'),
                    # No activation
                    keras_layers.Conv2DTranspose(
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

def build_network(input_hold, output_size=128, return_layers=False, keep_dim=4):
    svae = Spatial_VAE((39,39,6), input_hold, scope='spatial_vae', lr=1e-4)

    spatial_matrix = tf.stack(svae.z_list, axis=-1)  # (None, 128, 4)
    spatial_matrix = tf.stop_gradient(spatial_matrix)
    tvae = Temporal_VAE((128, 4), spatial_matrix, scope='temporal_vae', lr=1e-4)

    #feature = tf.concat([svae.z, tvae.z], axis=1) # (None, 192)
    feature = tf.concat([svae.z], axis=1) # (None, 192)

    train_ops = [svae.update, tvae.update]

    loss = {
            'svae': svae.elbo_loss,
            'tvae': tvae.elbo_loss
            }
    encoding_var = svae.trainable_variables + tvae.trainable_variables

    sampler = {
            'svae': svae.random_sample
            }

    return feature, train_ops, loss, encoding_var, sampler

if __name__ == '__main__':
    #data = np.random.sample((100,2))
    #iter = dataset.make_initializable_iterator() # create the iterator
    #el = iter.get_next()
    #with tf.Session() as sess:
    #    # feed the placeholder with data
    #    sess.run(iter.initializer, feed_dict={ x: data }) 
    #    print(sess.run(el)) # output [ 0.52374458  0.71968478]

    input_hold = tf.placeholder(dtype=tf.float32, shape=[None, 39, 39, 24])
    svae = Spatial_VAE((39,39,6), input_hold, scope='spatial_vae', lr=1e-4, num_stack=4)

    spatial_matrix = tf.stack(svae.z_list, axis=-1)  # (None, 128, 4)
    spatial_matrix = tf.stop_gradient(spatial_matrix)
    tvae = Temporal_VAE((128, 4), spatial_matrix, scope='temporal_vae', lr=1e-4)

    for layer in svae.inference_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)
    for layer in svae.generative_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)

    print(spatial_matrix.get_shape())

    for layer in tvae.inference_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)
    for layer in tvae.generative_net.layers:
        print(layer._name, layer.input_shape, layer.output_shape)

    for variable in svae.trainable_variables:
        print(variable)

    #for layer in tvae.inference_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)
    #for layer in tvae.generative_net.layers:
    #    print(layer._name, layer.input_shape, layer.output_shape)


#     
#     checkpoint_directory = "./tmp/training_checkpoints"
#     checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
# 
#     checkpoint = tf.train.Checkpoint()
# 
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         checkpoint.save(file_prefix=checkpoint_prefix, session=sess)
#         I = np.random.random(size=(2,39,39,24))
#         feed_dict = {input_ph:I}
#         update_feed_dict = {vae_train_pipe:np.random.random(size=(2,39,39,6))}
#         sess.run(vae.update, update_feed_dict)
#         sess.run(vae.update, update_feed_dict)
#         sess.run(vae.update, update_feed_dict)
#         file_writer = tf.summary.FileWriter(logdir='./tmp/logs/test', graph=sess.graph)
# 

    print('graph build done') 
