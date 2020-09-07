'''
- Self-play
- Subpolicy
- Shared CNN
- PPO
- No UAV
'''

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
import numpy as np
import random
import math
from collections import defaultdict

import policy

from utility.utils import MovingAverage as MA
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.ppo import PPO_multimodes as Network

assert len(sys.argv) == 4
device_t = sys.argv[3]

num_mode = 3
MODE_NAME = lambda mode: ['_attack', '_scout', '_defense', ''][mode]

setting_paths = ['setting_ppo_attacker.ini', 'setting_ppo_scout.ini', 'setting_ppo_defense.ini']
red_policies = [policy.Roomba, policy.Roomba, policy.AStar]

OVERRIDE = False
PROGBAR = False
LOG_DEVICE = False
RBETA = 0.8

## Training Directory Reset
TRAIN_NAME = sys.argv[2]+str(time.time()) #'fix_baseline'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.95
NENV = 8 #multiprocessing.cpu_count() // 2

## Data Path
path_create(LOG_PATH, override=OVERRIDE)
path_create(MODEL_PATH, override=OVERRIDE)
path_create(SAVE_PATH, override=OVERRIDE)

## Import Shared Training Hyperparameters
config = configparser.ConfigParser()
config.read('config.ini')

# Training
total_episodes = config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep = config.getint('TRAINING', 'MAX_STEP')

gamma = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd = config.getfloat('TRAINING', 'GAE_LAMBDA')
ppo_e = config.getfloat('TRAINING', 'PPO_EPSILON')
critic_beta = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta = config.getfloat('TRAINING', 'ENTROPY_BETA')

lr_a = config.getfloat('TRAINING', 'LR_ACTOR')
lr_c = config.getfloat('TRAINING', 'LR_CRITIC')

# Log Setting
save_network_frequency = config.getint('LOG', 'SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*4
moving_average_step = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame = config.getint('DEFAULT', 'KEEP_FRAME')
map_size = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 128
epoch = 2
minbatch_size = 2000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 7 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
global_episode_rewards = [MA(moving_average_step) for _ in range(num_mode)]
global_environment_rewards = [MA(moving_average_step) for _ in range(num_mode)]
global_length = [MA(moving_average_step) for _ in range(num_mode)]
global_succeed = [MA(moving_average_step) for _ in range(num_mode)]
global_episodes = 0

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
max_epsilon = 0.70; max_at = 150000
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

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size)

envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

sess = tf.Session(config=config)

with tf.device(device_t):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step, NENV)
    subtrain_step = [tf.Variable(0, trainable=False) for _ in range(num_mode)]
    subtrain_step_next = [tf.assign_add(step, NENV) for step in subtrain_step]
    network = Network(in_size=input_size, action_size=action_space, scope='main', sess=sess, num_mode=num_mode, model_path=MODEL_PATH)

def train(trajs, updater, bootstrap=0, epoch=epoch, batch_size=minibatch_size, **kwargv):
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
        updater(*mdp_tuple, **kwargv, idx=MODE)

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)
network.initiate(saver, MODEL_PATH)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
global_episodes = sess.run(global_step) # Reset the counter
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

def reward_shape(prev_red_alive, red_alive, done, idx=None, additional_reward=None):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    reward = []
    red_flags = envs.red_flag_captured()
    blue_flags = envs.blue_flag_captured()
    for i in range(NENV):
        if idx == 0:
            # Attack (C/max enemy)
            num_prev_enemy = sum(prev_red_alive[i])
            num_enemy = sum(red_alive[i])
            reward.append((num_prev_enemy - num_enemy)*0.25)
        if idx == 1:
            if red_flags[i]:
                reward.append(1)
            else:
                reward.append(0)
        if idx == 2:
            if blue_flags[i]:
                reward.append(-1)
            elif done[i]:
                reward.append(1)
            else:
                reward.append(0)
    if additional_reward is not None:
        return np.array(reward) + additional_reward
    else:
        return np.array(reward)

print('Training Initiated:')
def get_action(states):
    a1, v1, logits1 = network.run_network(states, MODE)
    actions = np.reshape(a1, [NENV, num_blue])
    return a1, v1, logits1, actions

batch = []
num_batch = 0
mode_changed = False
MODE = 0

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)
while True:
    if mode_changed:
        mode_changed = False
        MODE = (MODE + 1) % num_mode
    #MODE = np.argmin(sess.run(subtrain_step))
    mode_episode = sess.run(subtrain_step[MODE])

    log_on = interval_flag(mode_episode, save_stat_frequency, 'log{}'.format(MODE))
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    reload_on = False # interval_flag(global_episodes,selfplay_reload, 'reload')
    
    # Bootstrap
    s1 = envs.reset(
            config_path=setting_paths[MODE],
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=red_policies[MODE]
        )
    num_blue = len(envs.get_team_blue()[0])
    num_red = len(envs.get_team_red()[0])
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    episode_env_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
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

        if step == max_ep:
            done[:] = True
        reward = reward_shape(was_alive_red, is_alive_red, done, MODE)

        if step == max_ep:
            env_reward[:] = -1
        else:
            env_reward = (raw_reward - prev_rew - 0.01)/100


        for i in range(NENV): 
            if not was_done[i]:
                episode_rew[i] += reward[i]
                episode_env_rew[i] += env_reward[i]
    
        a1, v1, logits1, actions = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                reward_function = (RBETA) * reward[env_idx] + (1-RBETA) * env_reward[env_idx] 
                trajs[idx].append([s0[idx], a[idx], reward_function, v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break

    global_episodes += NENV
    sess.run(global_step_next)
    sess.run(subtrain_step_next[MODE])
    if PROGBAR:
        progbar.update(global_episodes)

    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])

    if num_batch >= minbatch_size:
        train(batch, network.update_global, 0, epoch, minibatch_size, writer=writer, log=log_image_on, global_episodes=global_episodes)
        batch = []
        num_batch = 0
        mode_changed = True

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    global_episode_rewards[MODE].extend(episode_rew.tolist())
    global_environment_rewards[MODE].extend(episode_env_rew.tolist())
    global_length[MODE].extend(steps)
    global_succeed[MODE].extend(envs.blue_win())


    if log_on:
        tag = 'baseline_training/'
        step = sess.run(subtrain_step[MODE])
        record({
            tag+'length'+MODE_NAME(MODE): global_length[MODE](),
            tag+'win-rate'+MODE_NAME(MODE): global_succeed[MODE](),
            tag+'reward'+MODE_NAME(MODE): global_episode_rewards[MODE](),
            tag+'env_reward'+MODE_NAME(MODE): global_environment_rewards[MODE](),
        }, writer, step)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

