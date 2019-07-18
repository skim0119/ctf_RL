'''
- Subpolicy
- 2 att, 1 nav, 1 def
- Shared CNN
- PPO
- No UAV
'''
import pickle
import os
import shutil
import configparser
import sys

import signal
import threading
import multiprocessing

import tensorflow as tf

import time
import gym
import gym_cap
import gym_cap.envs.const as CONST
import numpy as np
import random
import math
from collections import defaultdict

# the modules that you can use to generate the policy. 
import policy

# Data Processing Module
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards, interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.ppo import PPO_multimodes as Network

num_mode = 3
env_setting_path = 'setting_full.ini'

## Training Directory Reset
OVERRIDE = True
TRAIN_NAME = 'adapt_train/ppo_subp'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
GPU_CAPACITY = 0.95

NENV = multiprocessing.cpu_count()  

if OVERRIDE:
    MODEL_LOAD_PATH = './model/ppo_subp_robust/' # initialize values
else:
    MODEL_LOAD_PATH = MODEL_PATH

## Data Path
path_create(LOG_PATH, override=OVERRIDE)
path_create(MODEL_PATH, override=OVERRIDE)
path_create(SAVE_PATH, override=OVERRIDE)

## Import Shared Training Hyperparameters
config = configparser.ConfigParser()
config.read('config.ini')

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
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*16
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 128
epoch = 2
minbatch_size = 6000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
log_episodic_reward = MA(moving_average_step)
log_length = MA(moving_average_step)
log_winrate = MA(moving_average_step)

## Map Setting
map_dir = 'fair_map/'
map_list = [map_dir+'board{}.txt'.format(i) for i in range(1,5)]
max_epsilon = 0.55; max_at = 1
def smoothstep(x, lowx=0.0, highx=1.0, lowy=0, highy=1):
    x = (x-lowx) / (highx-lowx)
    if x < 0:
        val = 0
    elif x > 1:
        val = 1
    else:
        val = x * x * (3 - 2 * x)
    return val*(highy-lowy)+lowy
def use_this_map(x, max_episode, max_prob):
    prob = smoothstep(x, highx=max_episode, highy=max_prob)
    if np.random.random() < prob:
        return random.choice(map_list)
    else:
        return None

## Policy Setting
heur_policy_list = [policy.Patrol, policy.Roomba, policy.Defense, policy.Random, policy.AStar]
heur_weight = [1,1,1,1,1]
heur_weight = np.array(heur_weight) / sum(heur_weight)
def use_this_policy():
    return np.random.choice(heur_policy_list, p=heur_weight)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size,
	config_path=env_setting_path)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
subtrain_step = [tf.Variable(0, trainable=False) for _ in range(num_mode)]
subtrain_step_next = [tf.assign_add(step, NENV) for step in subtrain_step]
network = Network(in_size=input_size, action_size=action_space, scope='main', sess=sess, num_mode=num_mode)

global_episodes = 0
saver = tf.train.Saver(max_to_keep=3)
network.initiate(saver, MODEL_LOAD_PATH)
if OVERRIDE:
    sess.run(tf.assign(global_step, 0)) # Reset the counter
else:
    global_episodes = sess.run(global_step)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

def train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None, mode=None):
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
    for mdp_tuple in it:
        network.update_global(*mdp_tuple, global_episodes, writer, log, mode)

def reward_shape(prev_red_alive, red_alive, done, def_reward=0):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    r = []
    red_flags = envs.red_flag()
    blue_flags = envs.blue_flag()
    for i in range(NENV):
        # Attack (C/max enemy)
        num_prev_enemy = sum(prev_red_alive[i])
        num_enemy = sum(red_alive[i])
        r.append((num_prev_enemy - num_enemy)*0.25)
        r.append((num_prev_enemy - num_enemy)*0.25)
        # Scout
        if red_flags[i]:
            r.append(1)
        else:
            r.append(0)
        # Defense
        if blue_flags[i]:
            r.append(-1)
        else:
            r.append(0)
    return np.array(r)

print('Training Initiated:')
def get_action(states):
    a1, v1, logits1 = [], [], []
    a, v, logits = network.run_network(states, 0)
    a1.extend(a[:2]); v1.extend(v[:2]); logits1.extend(logits[:2])
    a, v, logits = network.run_network(states, 1)
    a1.extend(a[2:3]); v1.extend(v[2:3]); logits1.extend(logits[2:3])
    a, v, logits = network.run_network(states, 2)
    a1.extend(a[3:]); v1.extend(v[3:]); logits1.extend(logits[3:])
    actions = np.reshape(a1, [NENV, num_blue])
    return np.array(a1), np.array(v1), np.array(logits1), actions

batch_att = []
batch_sct = []
batch_def = []
num_batch_att = 0
num_batch_sct = 0
num_batch_def = 0
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    if global_episodes > 20000:
        env_setting_path = 'setting_partial.ini'
    s1 = envs.reset(
            config_path=env_setting_path,
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=use_this_policy()
        )
    a1, v1, logits1, actions = get_action(s1)

    # Rollout
    stime = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        env_reward = (raw_reward-prev_rew-0.01)/100.0

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True

        reward = reward_shape(was_alive_red, is_alive_red, done, env_reward)
        episode_rew += env_reward
    
        a1, v1, logits1, actions = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([s0[idx], a[idx], reward[idx]+env_reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break

    global_episodes += NENV
    sess.run(global_step_next)
    for i in range(num_mode):
        sess.run(subtrain_step_next[i])
    
    for i in range(NENV):
        batch_att.append(trajs[4*i+0])
        batch_att.append(trajs[4*i+1])
        batch_sct.append(trajs[4*i+2])
        batch_def.append(trajs[4*i+3])
        num_batch_att += len(trajs[4*i+0]) + len(trajs[4*i+1])
        num_batch_sct += len(trajs[4*i+2])
        num_batch_def += len(trajs[4*i+3])

    if num_batch_att >= minbatch_size:
        stime = time.time()
        train(batch_att, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=0)
        batch_att = []
        num_batch_att = 0
    if num_batch_sct >= minbatch_size:
        stime = time.time()
        train(batch_sct, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=1)
        batch_sct = []
        num_batch_sct = 0
    if num_batch_def >= minbatch_size:
        stime = time.time()
        train(batch_def, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=2)
        batch_def = []
        num_batch_def = 0

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    log_episodic_reward.append(np.mean(episode_rew))
    log_length.append(np.mean(steps))
    log_winrate.append(np.mean(envs.blue_win()))

    if log_on:
        step = sess.run(global_step)
        tag = 'adapt_train_log/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'reward': log_episodic_reward(),
        }, writer, step)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)
