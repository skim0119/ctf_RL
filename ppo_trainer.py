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

import gym_cap.heuristic as policy

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.ppo2 import PPO as Network

device_t = '/gpu:0'

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = sys.argv[3]
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.95

NENV = 8 # multiprocessing.cpu_count() // 2
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = sys.argv[2] # env_settings/setting_full.ini

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

## Import Shared Training Hyperparameters
config_path = sys.argv[1]
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
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ') * 4
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
log_traintime = MovingAverage(moving_average_step)

log_attack_reward = MovingAverage(moving_average_step)
log_scout_reward = MovingAverage(moving_average_step)
log_defense_reward = MovingAverage(moving_average_step)

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
max_epsilon = 0.70; max_at = total_episodes
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
    return lambda: gym.make(
            'cap-v0',
            map_size=map_size,
            config_path=env_setting_path
            )
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOG_DEVICE, allow_soft_placement=True)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_NAME)

sess = tf.Session(config=config)

with tf.device(device_t):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step, NENV)
    network = Network(input_shape=input_size, action_size=action_space, scope='main', sess=sess)

# Resotre / Initialize
global_episodes = 0
saver = tf.train.Saver(max_to_keep=3)
network.initiate(saver, MODEL_PATH)
if OVERRIDE:
    sess.run(tf.assign(global_step, 0)) # Reset the counter
else:
    global_episodes = sess.run(global_step)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)


### TRAINING ###
def train(trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
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
        network.update_network(*mdp_tuple, global_episodes, writer, log)

def get_action(states):
    a1, v1, logits1 = network.run_network(states)
    actions = np.reshape(a1, [NENV, num_blue])
    return a1, v1, logits1, actions

def reward_shape(prev_red_alive, red_alive, done):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    reward = []
    red_flags = envs.red_flag_captured()
    blue_flags = envs.blue_flag_captured()
    for i in range(NENV):
        possible_reward = []
        # Attack (C/max enemy)
        num_prev_enemy = sum(prev_red_alive[i])
        num_enemy = sum(red_alive[i])
        possible_reward.append((num_prev_enemy - num_enemy)*0.25)
        # Scout
        if red_flags[i]:
            possible_reward.append(1)
        else:
            possible_reward.append(0)
        # Defense
        if blue_flags[i]:
            possible_reward.append(-1)
        elif done[i]:
            possible_reward.append(1)
        else:
            possible_reward.append(0)

        reward.append(possible_reward)

    return np.array(reward)

batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    case_rew = [np.zeros(NENV) for _ in range(3)]
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    s1 = envs.reset(
            config_path=env_setting_path,
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=use_this_policy()
        )
    a1, v1, logits1, actions = get_action(s1)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1
        
        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        reward = (raw_reward - prev_rew - 0.01)/100.0
        if step == max_ep:
            reward[:] = -1
            done[:] = True
        episode_rew += reward

        shaped_reward = reward_shape(was_alive_red, is_alive_red, done)
        for i in range(NENV): 
            if not was_done[i]:
                for j in range(3):
                    case_rew[j][i] += shaped_reward[i,j]

        a1, v1, logits1, actions = get_action(s1)

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([s0[idx], a[idx], reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
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
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    
    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    log_attack_reward.extend(case_rew[0].tolist())
    log_scout_reward.extend(case_rew[1].tolist())
    log_defense_reward.extend(case_rew[2].tolist())

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        tag = 'baseline_training/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'redwin-rate': log_redwinrate(),
            tag+'env_reward': log_episodic_reward(),
            tag+'rollout_time': log_looptime(),
            tag+'train_time': log_traintime(),
            tag+'reward_attack': log_attack_reward(),
            tag+'reward_scout': log_scout_reward(),
            tag+'reward_defense': log_defense_reward(),
        }, writer, global_episodes)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)

