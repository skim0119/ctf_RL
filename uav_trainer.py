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

import time
import gym
import gym_cap
import numpy as np
import random
import math
from collections import defaultdict

import policy

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.ppo2 import PPO as Network

device_ground = '/gpu:0'
device_air = '/gpu:0'

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'UAV_TRAIN_SF_4'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.95

NENV = multiprocessing.cpu_count() // 2
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'uav_settings.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

## Import Shared Training Hyperparameters
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = 1000000#config.getint('TRAINING', 'TOTAL_EPISODES')
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
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ') // 2
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
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
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_explore = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
def use_fair_map():
    return random.choice(map_list)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make(
            'cap-v0',
            map_size=map_size,
            config_path=env_setting_path
            )
envs_list = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs_list, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOG_DEVICE, allow_soft_placement=True)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
with tf.device(device_ground):
    network = Network(input_shape=input_size, action_size=action_space, scope='ground', sess=sess)
with tf.device(device_air):
    network_air = Network(input_shape=input_size, action_size=action_space, scope='uav', sess=sess)

# Resotre / Initialize
global_episodes = 0
saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=4)
network.initiate(saver, MODEL_PATH)
if OVERRIDE:
    sess.run(tf.assign(global_step, 0)) # Reset the counter
else:
    global_episodes = sess.run(global_step)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes) # It save both network


### TRAINING ###
def train(nn, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        td_target, advantages = gae(traj[2], traj[3], 0,
                gamma, lambd, normalize=False)
        
        traj_buffer['state'].extend(traj[0])
        traj_buffer['action'].extend(traj[1])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)
        traj_buffer['logit'].extend(traj[4])

    if buffer_size < 10:
        return

    it = batch_sampler(
            batch_size,
            epoch,
            np.stack(traj_buffer['state']),
            np.stack(traj_buffer['action']),
            np.stack(traj_buffer['td_target']),
            np.stack(traj_buffer['advantage']),
            np.stack(traj_buffer['logit'])
        )
    i = 0
    for mdp_tuple in it:
        nn.update_network(*mdp_tuple, global_episodes, writer, log and (i==0))
        i+=1

def get_action(states):
    states_rsh = np.reshape(states, [NENV, num_blue+num_red]+input_size[1:])
    blue_air, blue_ground, red_air, red_ground = np.split(states_rsh, [2,6,8], axis=1)
    blue_states, red_states = np.split(states_rsh, [6], axis=1)

    # BLUE GET ACTION
    blue_air = np.reshape(blue_air, [NENV*2]+input_size[1:])
    blue_ground = np.reshape(blue_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground = network.run_network(blue_ground)
    action_air, value_air, logit_air = network_air.run_network(blue_air)

    action_rsh = np.concatenate([action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)

    # RED GET ACTION (Comment this section to make it single-side control and return blue_states)
    red_air = np.reshape(red_air, [NENV*2]+input_size[1:])
    red_ground = np.reshape(red_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground = network.run_network(red_ground)
    action_air, value_air, logit_air = network_air.run_network(blue_air)

    action_rsh = np.concatenate([action_rsh, action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value, value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit, logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)

    # RESHAPE
    action = action_rsh.reshape([-1])
    value = value.reshape([-1])
    logit = logit.reshape([NENV*12, 5])

    return states, action, value, logit, action_rsh


batch_ground, batch_air = [], []
num_batch = 0
while global_episodes < total_episodes:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    play_save_on = interval_flag(global_episodes, 5000, 'replay_save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    was_alive = [True for agent in range(NENV*(num_blue*num_red))]
    was_done = [False for env in range(NENV)]
    is_air = np.array([agent.is_air for agent in envs.get_team_blue().flat]).reshape([NENV, num_blue])
    is_air_red = np.array([agent.is_air for agent in envs.get_team_red().flat]).reshape([NENV, num_red])
    is_air = np.concatenate([is_air, is_air_red], axis=1).reshape([-1])

    trajs = [Trajectory(depth=5) for _ in range((num_blue+num_red)*NENV)] # Trajectory per agent
    
    # Bootstrap
    s1 = envs.reset(config_path=env_setting_path)
    s1, a1, v1, logits1, actions = get_action(s1)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1
        
        #actions = np.concatenate([actions, np.zeros_like(actions)], axis=1)
        #actions = np.concatenate([actions, np.random.randint(5, size=actions.shape)], axis=1)
        s1, reward, done, info = envs.step(actions)
        reward_red = np.array([i['red_reward'] for i in info])
        env_reward = np.vstack((reward, reward_red)).T.reshape([-1])

        is_alive = np.array([agent.isAlive for agent in envs.get_team_blue().flat]).reshape([NENV, num_blue])
        is_alive_red = np.array([agent.isAlive for agent in envs.get_team_red().flat]).reshape([NENV, num_red])
        is_alive = np.concatenate([is_alive, is_alive_red], axis=1).reshape([-1])


        episode_rew += reward * (1-np.array(was_done, dtype=int))

        s1, a1, v1, logits1, actions = get_action(s1)

        # push to buffer
        #TODO: completely separate the reward function for ground and air agents
        # For air : Give extra reward for flag (not sure how much), and extra reward for revealing the map
        #TODO: Reconsider parallelization construction
        # Problem arise when environment does not end simultaneously, and matrix operations are asynchronized due to it.

        was_done = np.array(was_done, dtype=bool)
        pre_air_reward = (np.isin(s1[::6,:,:,-12],-1).sum(axis=2).sum(axis=1) * (-0.1) + np.isin(s1[::6,:,:,-10], [1, -1]).astype(int).sum(axis=2).sum(axis=1)) * np.repeat(np.array(~was_done,dtype=int), 2)
        air_reward = (np.isin(s1[::6,:,:,-6],-1).sum(axis=2).sum(axis=1) * (-0.1) + np.isin(s1[::6,:,:,-4], [1, -1]).astype(int).sum(axis=2).sum(axis=1)) * np.repeat(np.array(~was_done,dtype=int), 2)
        for idx in range(NENV*(num_blue+num_red)):
        #for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // (num_blue+num_red)
            env_team_idx = idx // 6
            if was_alive[idx] and not was_done[env_idx]:
                if is_air[idx]:
                    #agent_reward = air_reward[env_team_idx] + env_reward[env_team_idx]
                    agent_reward = air_reward[env_team_idx] - pre_air_reward[env_team_idx]
                else:
                    agent_reward = env_reward[env_team_idx]
                trajs[idx].append([s0[idx], a[idx], agent_reward, v0[idx], logits[idx]])

        was_alive = is_alive
        was_done = done

        if np.all(done):
            explore_factor = np.mean(air_reward)
            break
    etime_roll = time.time()
            
    # Split air trajectory and ground trajectory
    for idx in range(NENV*(num_blue+num_red)):
        if is_air[idx]:
            batch_air.append(trajs[idx])
        else:
            batch_ground.append(trajs[idx])
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        train(network, batch_ground, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        train(network_air, batch_air, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch_ground, batch_air = [], []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    
    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)
    log_explore.append(explore_factor)

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        tag = 'uav_training/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'redwin-rate': log_redwinrate(),
            tag+'env_reward': log_episodic_reward(),
            tag+'rollout_time': log_looptime(),
            tag+'explore': log_explore(),
            tag+'train_time': log_traintime(),
        }, writer, global_episodes)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)

