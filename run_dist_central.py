import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import sys

import shutil
import configparser

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
TRAIN_NAME = 'C51_CENTRAL_02'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
GPU_CAPACITY = 0.95

NENV = multiprocessing.cpu_count()
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'env_settings_3v3_boot.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)

## Import Shared Training Hyperparameters
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep         = config.getint('TRAINING', 'MAX_STEP')
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
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 1#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 512
epoch = 10
minimum_batch_size = 2048
print(minimum_batch_size)

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None,20,20,6]

## Logger Initialization 
log_episodic_reward = MovingAverage(moving_average_step)
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size,
                            config_path=env_setting_path)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame=keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name='Dist C51 model (Central) : '+TRAIN_NAME)

network = Network(input_shape=input_size, action_size=action_space, scope='main', save_path=MODEL_PATH)

# Resotre / Initialize
global_episodes = 0
network.initiate()

writer = tf.summary.create_file_writer(LOG_PATH)
network.save(global_episodes)


### TRAINING ###
def train(trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)
        
        traj_buffer['state'].extend(traj[0])
        traj_buffer['reward'].extend(traj[1])
        traj_buffer['done'].extend(traj[2])
        traj_buffer['next_state'].extend(traj[3])

    if buffer_size < 10:
        return

    it = batch_sampler(
            batch_size,
            epoch,
            np.stack(traj_buffer['state']),
            np.stack(traj_buffer['reward']),
            np.stack(traj_buffer['done']),
            np.stack(traj_buffer['next_state']))
    losses = []
    for mdp_tuple in it:
        loss = network.update_network(*mdp_tuple, step)
        losses.append(loss)
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_critic_loss', np.mean(losses), step=step)
            writer.flush()

batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=4) for _ in range(NENV)]
    
    # Bootstrap
    s1 = envs.reset(config_path=env_setting_path,
                    policy_red=policy.Roomba,
                    policy_blue=policy.Roomba)
    s1 = envs.get_obs_blue()

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep+1):
        s0 = s1
        
        s1, reward, done, info = envs.step()
        s1 = envs.get_obs_blue()
        episode_rew += reward

        # push to buffer
        for env_idx in range(NENV):
            if not was_done[env_idx]:
                trajs[env_idx].append([
                    s0[env_idx],
                    reward[env_idx],
                    done[env_idx],
                    s1[env_idx]])

        was_done = done

        if np.all(done):
            break
    etime_roll = time.time()
            
    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)

    steps = []
    for env_id in range(NENV):
        steps.append(len(trajs[env_id]))
    
    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        with writer.as_default():
            tag = 'baseline_training/'
            tf.summary.scalar(tag+'length', log_length(), step=global_episodes)
            tf.summary.scalar(tag+'win-rate', log_winrate(), step=global_episodes)
            tf.summary.scalar(tag+'redwin-rate', log_redwinrate(), step=global_episodes)
            tf.summary.scalar(tag+'env_reward', log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag+'rollout_time', log_looptime(), step=global_episodes)
            tf.summary.scalar(tag+'train_time', log_traintime(), step=global_episodes)
            writer.flush()
        
    if save_on:
        network.save(global_episodes)
