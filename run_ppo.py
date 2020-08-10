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
from utility.logger import *
from utility.gae import gae

from method.ActorCritic import PPO_Module as Network

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'PPO_STACK_Full_01_convoy' 
TRAIN_TAG = 'PPO e2e model w Stacked Frames: '+TRAIN_NAME
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
MAP_PATH = './fair_3g_20'
GPU_CAPACITY = 0.95

NENV = multiprocessing.cpu_count() // 4
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'env_setting_3v3_3g_full_convoy.ini'

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
save_stat_frequency    = 128#config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = 128#config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 1#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 4096
print(minimum_batch_size)

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

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
num_agent = num_blue# + num_red

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

network = Network(input_shape=input_size, action_size=action_space, scope='main', save_path=MODEL_PATH)

# Resotre / Initialize
global_episodes = network.initiate()
print(global_episodes)

writer = tf.summary.create_file_writer(LOG_PATH)
network.save(global_episodes)


### TRAINING ###
def train(network, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    advantage_list = []
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        td_target, advantages = gae(traj[2], traj[3], traj[5][-1],
                gamma, lambd, normalize=False)
        advantage_list.append(advantages)
        
        traj_buffer['state'].extend(traj[0])
        traj_buffer['action'].extend(traj[1])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)
        traj_buffer['old_log_logit'].extend(traj[4])

    train_dataset = tf.data.Dataset.from_tensor_slices({
        'state': np.stack(traj_buffer['state']),
        'old_log_logit': np.stack(traj_buffer['old_log_logit']),
        'action': np.stack(traj_buffer['action']),
        'advantage': np.stack(traj_buffer['advantage']).astype(np.float32),
        'td_target': np.stack(traj_buffer['td_target']).astype(np.float32),
        }).shuffle(64).repeat(epoch).batch(batch_size)

    logs = network.update_network(train_dataset, log=log)
    if log:
        with writer.as_default():
            tag = 'summary/'
            tb_log_histogram(np.array(advantage_list), tag+'dec_advantages', step=global_episodes)
            for name, val in logs.items():
                tf.summary.scalar(tag+name, val, step=step)
            writer.flush()

def get_action(states):
    a1, v1, p1 = network.run_network(states)
    actions = np.reshape(a1, [NENV, num_blue])
    return a1, v1, p1, actions

batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=6) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    s1 = envs.reset(
            map_pool=map_list,
            config_path=env_setting_path,
            policy_red=policy.Roomba)
    s1 = s1.astype(np.float32)
    a1, v1, p1, actions = get_action(s1)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep):
        s0 = s1
        a0, v0 = a1, v1
        p0 = p1 
        
        s1, reward, done, info = envs.step(actions)
        s1 = s1.astype(np.float32)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        episode_rew += reward

        a1, v1, p1, actions = get_action(s1)

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            #if was_alive[idx] and not was_done[env_idx]:
            if not was_done[env_idx]:
                trajs[idx].append([s0[idx], a0[idx], reward[env_idx], v0[idx], p0[idx], v1[idx]])

        was_alive = is_alive
        was_done = done

        if np.all(done):
            break
    etime_roll = time.time()
            
    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
        train(network, batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)
    
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
