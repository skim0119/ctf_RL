import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import sys

import shutil
import configparser
import argparse

import signal
import threading
import multiprocessing

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import gym
import gym_cap
import numpy as np
import random
import math
from collections import defaultdict

import gym_cap.heuristic as policy

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.logger import record
from utility.gae import gae

from method.dist import SFK as Network

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'DIST_SFK_01'
TRAIN_TAG = 'Central value decentralized control, '+TRAIN_NAME
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
MAP_PATH = './fair_3g_40'
GPU_CAPACITY = 0.95

NENV = multiprocessing.cpu_count()
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'env_setting_3v3_3g_partial.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)

## Import Shared Training Hyperparameters
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep         = 200#config.getint('TRAINING', 'MAX_STEP')
gamma          = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd          = config.getfloat('TRAINING', 'GAE_LAMBDA')
ppo_e          = config.getfloat('TRAINING', 'PPO_EPSILON')
critic_beta    = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta   = config.getfloat('TRAINING', 'ENTROPY_BETA')
lr_a           = config.getfloat('TRAINING', 'LR_ACTOR')
lr_c           = config.getfloat('TRAINING', 'LR_CRITIC')

# Log Setting
save_network_frequency = config.getint('LOG', 'SAVE_NETWORK_FREQ')
save_stat_frequency    = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = 39#config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 1#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = 40#config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 1024 * 4
print(minimum_batch_size)

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]
cent_input_size = [None, map_size, map_size, nchannel]

## Logger Initialization 
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH) if path[:5]=='board']
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size,
                            config_path=env_setting_path)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame=keep_frame, size=vision_dx)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])
num_agent = num_blue#+num_red

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

atoms = 8
network = Network(
        central_obs_shape=cent_input_size,
        decentral_obs_shape=input_size,
        action_size=action_space, 
        atoms=atoms,
        save_path=MODEL_PATH)

# Resotre / Initialize
global_episodes = 0
network.initiate()

writer = tf.summary.create_file_writer(LOG_PATH)
network.save(global_episodes)


### TRAINING ###
def train_central(network, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        #reward = traj[1]
        #critic = traj[6].tolist()
        phi = np.array(traj[7]).tolist()
        psi = np.array(traj[8]).tolist()
        
        #_, advantages = gae(reward, critic, 0,
        #        gamma, lambd, normalize=False)
        td_target, _ = gae(phi, psi, np.zeros_like(phi[0]),
                gamma, lambd, normalize=False)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['reward'].extend(traj[1])
        traj_buffer['done'].extend(traj[2])
        traj_buffer['next_state'].extend(traj[3])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['b_mean'].extend(traj[4])
        traj_buffer['b_log_var'].extend(traj[5])
        traj_buffer['next_mean'].extend(traj[9])
        traj_buffer['next_log_var'].extend(traj[10])

    if buffer_size < 10:
        return

    it = batch_sampler(
            batch_size,
            epoch,
            np.stack(traj_buffer['state']),
            np.stack(traj_buffer['b_mean']),
            np.stack(traj_buffer['b_log_var']),
            np.stack(traj_buffer['td_target']),
            np.stack(traj_buffer['next_mean']),
            np.stack(traj_buffer['next_log_var']),
            )
    psi_losses = []
    elbo_losses = []
    kalman_losses = []
    for mdp in it:
        info = network.update_central_critic(*mdp)
        if log:
            psi_losses.append(info['psi_mse'])
            elbo_losses.append(info['elbo'])
            kalman_losses.append(info['pred_loss'])
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_psi_loss', np.mean(psi_losses), step=step)
            tf.summary.scalar(tag+'main_ELBO_loss', np.mean(elbo_losses), step=step)
            tf.summary.scalar(tag+'main_Kalman_loss', np.mean(kalman_losses), step=step)
            writer.flush()

def train_decentral(network, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        reward = traj[3]
        critic = traj[6]
        phi = np.array(traj[8]).tolist()
        psi = np.array(traj[9]).tolist()
        
        _, advantages = gae(reward, critic, 0,
                gamma, lambd, normalize=False)
        td_target, _ = gae(phi, psi, np.zeros_like(phi[0]),
                gamma, lambd, normalize=False)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['belief'].extend(traj[1])
        traj_buffer['logit'].extend(traj[7])
        traj_buffer['action'].extend(traj[2])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)

    if buffer_size < 10:
        return

    it = batch_sampler(batch_size, epoch,
                       np.stack(traj_buffer['state']),
                       np.stack(traj_buffer['belief']),
                       np.stack(traj_buffer['logit']),
                       np.stack(traj_buffer['action']),
                       np.stack(traj_buffer['td_target']),
                       np.stack(traj_buffer['advantage']),)
    actor_losses = []
    dec_psi_losses = []
    entropy = []
    for mdp in it:
        info = network.update_ppo(*mdp)
        if log:
            actor_losses.append(info['actor_loss'])
            dec_psi_losses.append(info['psi_loss'])
            entropy.append(info['entropy'])
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'dec_actor_loss', np.mean(actor_losses), step=step)
            tf.summary.scalar(tag+'dec_psi_loss', np.mean(dec_psi_losses), step=step)
            tf.summary.scalar(tag+'dec_entropy', np.mean(entropy), step=step)
            writer.flush()

def train_reward_prediction(network, traj, epoch, batch_size, writer=None, log=False, step=None):
    buffer_size = len(traj)
    if buffer_size < 10:
        return
    reward_stack = np.stack(traj[1])
    it = batch_sampler(batch_size, epoch,
                       np.stack(traj[0]),
                       np.stack(traj[1]),
                       np.stack(traj[2]),
                       np.stack(traj[3]))
    reward_losses = []
    for mdp_tuple in it:
        info = network.update_reward_prediction(*mdp_tuple)
        reward_losses.append(info['reward_mse'])
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_reward_loss', np.mean(reward_losses), step=step)
            writer.flush()

def train_ma_value(network, trajs, bootstrap=0.0, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)
        
        traj_buffer['state'].extend(traj[0])
        traj_buffer['belief'].extend(traj[1])
        traj_buffer['value_central'].extend(traj[2])
        traj_buffer['mask'].extend(traj[3])

    if buffer_size < 10:
        return

    it = batch_sampler(200, 1,
                       np.stack(traj_buffer['state']),
                       np.stack(traj_buffer['belief']),
                       np.stack(traj_buffer['value_central']),
                       np.stack(traj_buffer['mask']),)
    losses = []
    for mdp in it:
        info = network.update_multiagent_critic(*mdp)
        if log:
            losses.append(info['ma_critic'])
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'dec_ma_critic', np.mean(losses), step=step)
            writer.flush()

def get_action(log_logits):
    a = tf.random.categorical(log_logits, 1, dtype=tf.int32).numpy().ravel()
    action = np.reshape(a, [NENV, num_agent])
    return a, action

reward_training_buffer = Trajectory(depth=4) # Centralized
batch = []
dec_batch = []
ma_batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    is_alive = [True for agent in envs.get_team_blue().flat]
    is_done = [False for env in range(NENV*num_agent)]

    trajs = [Trajectory(depth=10) for _ in range(NENV*num_agent)]
    ma_trajs = [Trajectory(depth=4) for _ in range(NENV)]
    cent_trajs = [Trajectory(depth=11) for _ in range(NENV)]
    bmean1 = np.zeros([NENV, atoms], dtype=np.float32)
    blogvar1 = np.zeros([NENV, atoms], dtype=np.float32)
    
    # Bootstrap
    s1 = envs.reset(
            map_pool=map_list,
            config_path=env_setting_path,
            policy_red=policy.Roomba,
            #policy_blue=policy.Roomba,
            mode='continue')
    s1 = s1.astype(np.float32)
    cent_s1 = envs.get_obs_blue().astype(np.float32) # Centralized

    env_critic, env_feature, env_pred_feature = network.run_network_central(cent_s1, bmean1, blogvar1) 
    belief = env_feature['latent'].numpy()
    belief = np.repeat(belief, num_agent, axis=0)
    actor, critic = network.run_network_decentral(s1, belief)
    a1, action = get_action(actor['log_softmax'].numpy())

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep):
        s0 = s1
        a0 = a1
        v0 = critic['critic'].numpy()[:,0]
        psi0 = critic['psi'].numpy()
        log_logits0 = actor['log_softmax'].numpy()
        cent_s0 = cent_s1
        bmean0, blogvar0 = bmean1, blogvar1
        cent_v0 = env_critic['critic'].numpy()[:,0]
        cent_psi = env_critic['psi'].numpy()
        cent_pmean = env_pred_feature['pred_z_mean'].numpy()
        cent_plogvar = env_pred_feature['pred_z_log_var'].numpy()
        was_alive = is_alive
        was_done = is_done
        
        # Run environment
        s1, reward, done, info = envs.step(action)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        s1 = s1.astype(np.float32) # Decentralize observation
        cent_s1 = envs.get_obs_blue().astype(np.float32) # Centralized
        episode_rew += reward

        # Run central network
        env_critic, env_feature, env_pred_feature = network.run_network_central(cent_s1, bmean0, blogvar0) 
        cent_phi1 = env_critic['phi'].numpy()
        bmean1 = env_feature['z_mean'].numpy() # stop gradient
        blogvar1 = env_feature['z_log_var'].numpy()
        belief = env_feature['latent'].numpy()
        belief = np.repeat(belief, num_agent, axis=0)
        # Run decentral network
        actor, critic = network.run_network_decentral(s1, belief)
        a1, action = get_action(actor['log_softmax'])
        phi1 = critic['phi'].numpy()
        reward_pred1 = critic['reward_predict'].numpy()[:,0]

        # Buffer
        for idx in range(NENV*num_agent):
            env_idx = idx // num_agent
            if not was_done[env_idx] and was_alive[idx]:
                # Decentral trajectory
                trajs[idx].append([
                    s0[idx],
                    belief[idx],
                    a0[idx],
                    reward_pred1[idx], # Advantage
                    done[env_idx],
                    s1[idx],
                    v0[idx], # Advantage
                    log_logits0[idx], # PPO
                    phi1[idx], # phi: one-step ahead
                    psi0[idx],
                    ])
        for env_idx in range(NENV):
            if not was_done[env_idx]:
                # Reward training buffer
                reward_training_buffer.append([
                    cent_s0[env_idx],
                    reward[env_idx],
                    bmean0[env_idx],
                    blogvar0[env_idx]])
                # Central trajectory
                cent_trajs[env_idx].append([
                    cent_s0[env_idx],
                    reward[env_idx],
                    done[env_idx],
                    cent_s1[env_idx],
                    bmean0[env_idx],
                    blogvar0[env_idx],
                    cent_v0[env_idx],
                    cent_phi1[env_idx],
                    cent_psi[env_idx],
                    cent_pmean[env_idx],
                    cent_plogvar[env_idx],])
                # MA trajectory
                ma_trajs[env_idx].append([
                    s0[env_idx*num_agent:(env_idx+1)*num_agent],
                    belief[env_idx*num_agent:(env_idx+1)*num_agent],
                    cent_v0[env_idx],
                    is_alive[env_idx*num_agent:(env_idx+1)*num_agent],])

        if np.all(done):
            break
    etime_roll = time.time()

    # Trajectory Training
    '''
    for idx in range(NENV*num_agent):
        if is_air[idx]:
            batch.append(trajs[idx])
        else:
            batch.append(trajs[idx])
    '''
    batch.extend(cent_trajs)
    dec_batch.extend(trajs)
    ma_batch.extend(ma_trajs)
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
        train_central(network, batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        train_decentral(network, dec_batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        train_ma_value(network, ma_batch, 0, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch = []
        dec_batch = []
        ma_batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)
    # Reward Training
    if len(reward_training_buffer) > 4096:
        log_rt_on = interval_flag(global_episodes, save_image_frequency, 'rt_log')
        train_reward_prediction(network, reward_training_buffer, epoch=2, batch_size=64, writer=writer, log=log_rt_on, step=global_episodes)
        #reward_training_buffer.clear()
        temp_buffer = Trajectory(depth=4)
        for i in range(len(reward_training_buffer)):
            r = [col[i] for col in reward_training_buffer.buffer]
            if r[1] != 0:
                temp_buffer.append(r)
        reward_training_buffer.clear()
        reward_training_buffer = temp_buffer
        if len(reward_training_buffer) > 8000: # Purge
            reward_training_buffer.clear()
    
    log_episodic_reward.extend(episode_rew.tolist())
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    if PROGBAR:
        progbar.update(global_episodes)

    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    if log_on:
        with writer.as_default():
            tag = 'baseline_training/'
            tf.summary.scalar(tag+'win-rate', log_winrate(), step=global_episodes)
            tf.summary.scalar(tag+'redwin-rate', log_redwinrate(), step=global_episodes)
            tf.summary.scalar(tag+'env_reward', log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag+'rollout_time', log_looptime(), step=global_episodes)
            tf.summary.scalar(tag+'train_time', log_traintime(), step=global_episodes)
            writer.flush()
        
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    if save_on:
        network.save(global_episodes)
