import tensorflow as tf
import os
import sys
import time
import multiprocessing
from utility.multiprocessing import SubprocVecEnv
import gym
import gym_cap
import policy

import numpy as np


BATCH_SIZE = 100

mirrored_strategy = tf.distribute.MirroredStrategy()

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential( [
                tf.keras.layers.InputLayer(input_shape=(39, 39, 6)),
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
                tf.keras.layers.Dense(units=9*9*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(9, 9, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
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
                    filters=6, kernel_size=3, strides=(1, 1), padding="SAME"),
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

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

latent_dim = 128

model = CVAE(latent_dim)

NENV = multiprocessing.cpu_count()

def make_env(map_size):
    return lambda: gym.make('cap-v0')
envs = [make_env(20) for i in range(NENV)]
envs = SubprocVecEnv(envs, 1)
envs.reset(policy_red=policy.Roomba, policy_blue=policy.Roomba)

stime = time.time()
train_images = []
while len(train_images) < 40000:
    s1 = envs.reset(policy_red=policy.Roomba, policy_blue=policy.Roomba)
    trajs = [[] for _ in range(4*NENV)]
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]
    for step in range(150):
        s0 = s1
        s1,_,done,_ = envs.step()
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // 4
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append(s0[idx])
        was_alive = is_alive
        was_done = done

        if np.all(done):
            break
    for traj in trajs:
        train_images.extend(traj)

train_images = np.array(train_images).astype(np.float32)
print(train_images.shape)
mn, mx = train_images.min(axis=(1,2), keepdims=True), train_images.max(axis=(1,2), keepdims=True)
dif = mx-mn
dif[np.isclose(dif, 0)] = 1
train_images = (train_images - mn)/dif
print(train_images.min(), train_images.max())

print('{} sec elapsed to gather training dataset'.format(time.time() - stime))
print('{} length'.format(len(train_images)))
print('{} bit'.format(sys.getsizeof(train_images)))

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)

epochs = 1000
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for train_x in train_dataset:
             loss(compute_loss(model, train_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, '
            'time elapse for current epoch {}'.format(epoch, elbo, end_time - start_time))

