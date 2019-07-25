import tensorflow as tf

import os
import sys

import time
import multiprocessing
from utility.multiprocessing import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imageio

import gym
import gym_cap

import policy

from network.model_V3 import Spatial_VAE
from utility.utils import path_create
from utility.dataModule import oh_to_rgb

path = 'vae_test'
path_create(path)

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
# mn, mx = train_images.min(axis=(1,2), keepdims=True), train_images.max(axis=(1,2), keepdims=True)
# dif = mx-mn
# dif[np.isclose(dif, 0)] = 1
# train_images = (train_images - mn)/dif

print('{} sec elapsed to gather training dataset'.format(time.time() - stime))
print('{} length'.format(len(train_images)))
print('{} bit'.format(sys.getsizeof(train_images)))

#train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
batch_size = 128
num_batch = len(train_images)//batch_size
train_dataset = np.split(train_images[:num_batch*batch_size], len(train_images)//batch_size)

vae_train_pipe = tf.placeholder(tf.float32, shape=[None,39,39,6])
vae = Spatial_VAE((39,39,6), vae_train_pipe, scope='vae', lr=1e-4)
train_ops = vae.update

def generate_and_save_images(sess, network, epoch):
    predictions = sess.run(network.random_sample)
    fig = plt.figure(figsize=(4,6))

    #images = oh_to_rgb(predictions)

    for i in range(predictions.shape[0]):
        for j in range(6):
            plt.subplot(4, 6, i*6+j+1)
            plt.imshow(predictions[i,:,:,j], cmap='gray')
            #plt.imshow(images[i])
            plt.axis('off')

    plt.savefig(path+'/image_at_epoch_{:04d}.png'.format(epoch))

epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            sess.run(train_ops, feed_dict={vae_train_pipe: train_x})
            #gradients, loss = compute_gradients(model, train_x)
            #apply_gradients(optimizer, gradients, model.trainable_variables)
        end_time = time.time()

        if epoch % 10 == 0:
            generate_and_save_images(sess, vae, epoch)
            loss = sess.run(vae.elbo_loss, feed_dict={vae_train_pipe: train_x})
            #loss = tf.keras.metrics.Mean()
            #for train_x in train_dataset:
            #     loss(compute_loss(model, train_x))
            #elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '
                'time elapse for current epoch {}'.format(epoch, loss, end_time - start_time))
            

