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

from method.dist import DistCriticCentral as Network

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'DIST_SF_PASV_03'  # distributional kalman on V passive learning
TRAIN_TAG = 'Dist model SF : '+TRAIN_NAME
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
save_image_frequency   = 4#config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = 39#config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 1#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = 40#config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 4096
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
cent_network = Network(input_shape=cent_input_size, action_size=action_space, atoms=atoms, scope='main', save_path=MODEL_PATH+'/cent')

# Resotre / Initialize
global_episodes = 0
cent_network.initiate()

writer = tf.summary.create_file_writer(LOG_PATH)
cent_network.save(global_episodes)


### TRAINING ###
def train(network, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        states = np.array(traj[0])
        critic, _, _, _, phi, _, psi = network.run_network(states)
        critic = critic[:,0].numpy().tolist()
        phi = phi[:].numpy().tolist()
        psi = psi[:].numpy().tolist()
        
        _, advantages = gae(traj[1], critic, 0,
                gamma, lambd, normalize=False)
        td_target, _ = gae(phi, psi, np.zeros_like(phi[0]),
                gamma, lambd, normalize=False)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['reward'].extend(traj[1])
        traj_buffer['done'].extend(traj[2])
        traj_buffer['next_state'].extend(traj[3])
        traj_buffer['td_target'].extend(td_target)

    if buffer_size < 10:
        return

    it = batch_sampler(
            batch_size,
            epoch,
            np.stack(traj_buffer['state']),
            np.stack(traj_buffer['reward']),
            np.stack(traj_buffer['done']),
            np.stack(traj_buffer['next_state']),
            np.stack(traj_buffer['td_target']),
            )
    psi_losses = []
    elbo_losses = []
    for mdp in it:
        psi_losses.append(network.update_sf(mdp[0], mdp[4]))
        elbo_losses.append(network.update_decoder(mdp[0]))
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_critic_loss', np.mean(psi_losses), step=step)
            tf.summary.scalar(tag+'main_ELBO_loss', np.mean(elbo_losses), step=step)
            writer.flush()

def train_reward_prediction(network, traj, epoch, batch_size, writer=None, log=False, step=None):
    buffer_size = len(traj)
    if buffer_size < 10:
        return
    reward_stack = np.stack(traj[1])
    print(np.unique(reward_stack, return_counts=1))
    it = batch_sampler(batch_size, epoch,
                       np.stack(traj[0]),
                       np.stack(traj[1]))
    reward_losses = []
    for mdp_tuple in it:
        reward_loss = network.update_reward_prediction(*mdp_tuple)
        reward_losses.append(reward_loss)
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_reward_loss', np.mean(reward_losses), step=step)
            writer.flush()

reward_training_buffer = Trajectory(depth=2) # Centralized
batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    was_done = [False for env in range(NENV*num_agent)]

    #trajs = [Trajectory(depth=4) for _ in range(NENV*num_agent)]
    cent_trajs = [Trajectory(depth=4) for _ in range(NENV)]
    
    # Bootstrap
    s1 = envs.reset(
            map_pool=map_list,
            config_path=env_setting_path,
            policy_red=policy.Roomba,
            policy_blue=policy.Roomba,
            mode='continue')
    s1.astype(np.float32)
    is_air = np.array([agent.is_air for agent in envs.get_team_blue().flat])#.reshape([NENV, num_blue])
    cent_s1 = envs.get_obs_blue() # Centralized

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep):
        s0 = s1
        cent_s0 = cent_s1
        
        s1, reward, done, info = envs.step()
        s1.astype(np.float32)
        cent_s1 = envs.get_obs_blue() # Centralized
        episode_rew += reward

        # push to buffer
        '''
        for idx in range(NENV*num_agent):
            env_idx = idx // num_agent
            if not was_done[env_idx]:
                trajs[idx].append([
                    s0[idx],
                    reward[env_idx],
                    done[env_idx],
                    s1[idx]])
        '''
        for env_idx in range(NENV):
            if not was_done[env_idx]:
                reward_training_buffer.append([
                    cent_s0[env_idx],
                    reward[env_idx]])
                cent_trajs[env_idx].append([
                    cent_s0[env_idx],
                    reward[env_idx],
                    done[env_idx],
                    cent_s1[env_idx]])

        was_done = done

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
    num_batch += sum([len(traj) for traj in cent_trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
        train(cent_network, batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)
    # Reward Training
    if len(reward_training_buffer) > 4096:
        log_rt_on = interval_flag(global_episodes, save_image_frequency, 'rt_log')
        train_reward_prediction(cent_network, reward_training_buffer, epoch=2, batch_size=64, writer=writer, log=log_rt_on, step=global_episodes)
        #reward_training_buffer.clear()
        temp_buffer = Trajectory(depth=2)
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
        cent_network.save(global_episodes)
