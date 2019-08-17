import pickle
import os
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

from method.ppo import PPO_multimodes as Network

assert len(sys.argv) == 2

LOGDEVICE = False
PROGBAR = False
TRAIN_SUBP = False

num_mode = 3

## Training Directory Reset
TRAIN_NAME = sys.argv[1]
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.90
NENV = multiprocessing.cpu_count()  

MODEL_LOAD_PATH = './model/fix_baseline2'
ENV_SETTING_PATH = 'setting_full.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

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
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*4
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
batch_memory_size = 4000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 7 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
log_episodic_reward = MovingAverage(moving_average_step)
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
max_epsilon = 0.00; max_at = 1
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
	config_path=ENV_SETTING_PATH)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOGDEVICE)

sess = tf.Session(config=config)

global_episodes = 0
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
network = Network(in_size=input_size, action_size=action_space, sess=sess, num_mode=num_mode, scope='main')
saver = tf.train.Saver(max_to_keep=3, var_list=network.get_vars+[global_step])

# Resotre / Initialize
pretrained_vars = []
pretrained_vars_name = []
for varlist in network.a_vars[:-1]+network.c_vars[:-1]:
    for var in varlist:
        if var.name in pretrained_vars_name:
            continue
        pretrained_vars_name.append(var.name)
        pretrained_vars.append(var)
restoring_saver = tf.train.Saver(max_to_keep=3, var_list=pretrained_vars)
network.initiate(restoring_saver, MODEL_LOAD_PATH)
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

def meta_train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
    traj_buffer = defaultdict(list)
    sub_traj_buffer = [defaultdict(list) for _ in range(num_mode)]
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        # Meta Trajectory
        td_target, advantages = gae(traj[2], traj[3], 0,
                gamma, lambd, normalize=False)
        
        traj_buffer['state'].extend(traj[0])
        traj_buffer['action'].extend(traj[1])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)
        traj_buffer['logit'].extend(traj[4])

        # Subp Trajectory
        if TRAIN_SUBP:
            sub_traj = Trajectory(depth=5)
            mode = traj[1][0] # Initial mode
            for i in range(len(traj)):
                if traj[1][i] != mode:
                    sub_bootstrap = traj[9][i] # Next critic
                    td_target, advantages = gae(sub_traj[2], sub_traj[3], sub_bootstrap,
                            gamma, lambd, normalize=False)
                    sub_traj_buffer[mode]['state'].extend(sub_traj[0])
                    sub_traj_buffer[mode]['action'].extend(sub_traj[1])
                    sub_traj_buffer[mode]['td_target'].extend(td_target)
                    sub_traj_buffer[mode]['advantage'].extend(advantages)
                    sub_traj_buffer[mode]['logit'].extend(sub_traj[4])

                    sub_traj.clear()
                    mode = traj[1][i]

                sub_traj.append([
                        traj[0][i], # State
                        traj[6][i], # Action
                        traj[10][i]+traj[2][i], # Reward (indiv + shared)
                        traj[7][i], # Critic
                        traj[8][i] # Prob
                    ])
            td_target, advantages = gae(sub_traj[2], sub_traj[3], 0,
                    gamma, lambd, normalize=False)
            sub_traj_buffer[mode]['state'].extend(sub_traj[0])
            sub_traj_buffer[mode]['action'].extend(sub_traj[1])
            sub_traj_buffer[mode]['td_target'].extend(td_target)
            sub_traj_buffer[mode]['advantage'].extend(advantages)
            sub_traj_buffer[mode]['logit'].extend(sub_traj[4])

    if buffer_size < 10:
        return

    # Train Meta
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
        network.update_bandit(*mdp_tuple, global_episodes, writer, log)

    # Train Sub
    if TRAIN_SUBP:
        for mode in range(num_mode):
            it = batch_sampler(
                    batch_size,
                    epoch,
                    np.stack(sub_traj_buffer[mode]['state']),
                    np.stack(sub_traj_buffer[mode]['action']),
                    np.stack(sub_traj_buffer[mode]['td_target']),
                    np.stack(sub_traj_buffer[mode]['advantage']),
                    np.stack(sub_traj_buffer[mode]['logit'])
                )
            for mdp_tuple in it:
                network.update_global(*mdp_tuple, global_episodes, writer, log, mode)

def reward_shape(prev_red_alive, red_alive, done):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    reward_list = []
    red_flags = envs.red_flag_captured()
    blue_flags = envs.blue_flag_captured()
    for i in range(NENV):
        reward = np.array([0]*3)
        # Attack (C/max enemy)
        num_prev_enemy = sum(prev_red_alive[i])
        num_enemy = sum(red_alive[i])
        reward[0] = (num_prev_enemy - num_enemy)*0.25
        #Sct
        if red_flags[i]:
            reward[1] = 1
        else:
            reward[1] = 0
        #Def
        if blue_flags[i]:
            reward[2] = -1
        elif done[i]:
            reward[2] = 1
        else:
            reward[2] = 0
        reward_list.append(reward)
    return np.array(reward_list)

print('Training Initiated:')
def get_action(states, initial=False):
    if initial:
        network.initiate_confid(NENV*num_blue)
    action, critic, logits, bandit_action, bandit_critic, bandit_logit = network.run_network_with_bandit(states, True)

    actions = np.reshape(action, [NENV, num_blue])

    return bandit_action, bandit_critic, bandit_logit, actions, action, critic, logits

batch = []
num_batch = 0
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    reload_on = False # interval_flag(global_episodes,selfplay_reload, 'reload')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=11) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    s1 = envs.reset(
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=use_this_policy()
        )
    a1, v1, logits1, actions, sub_a1, sub_v1, sub_logits1 = get_action(s1, initial=True)

    # Rollout
    cumul_reward = np.zeros(NENV)
    for step in range(max_ep+1):
        s0 = s1
        a0, v0 = a1, v1
        logits0 = logits1
        sub_a0, sub_v0 = sub_a1, sub_v1
        sub_logits0 = sub_logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        env_reward = (raw_reward - prev_rew - 0.01)/100
        episode_rew += env_reward

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True

        task_reward = reward_shape(was_alive_red, is_alive_red, done)
    
        a1, v1, logits1, actions, sub_a1, sub_v1, sub_logits1 = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0
                sub_v1[idx] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([
                    s0[idx],
                    a0[idx],
                    env_reward[env_idx],
                    v0[idx],
                    logits0[idx],
                    v1[idx],
                    sub_a0[idx],
                    sub_v0[idx],
                    sub_logits0[idx],
                    sub_v1[idx],
                    task_reward[env_idx][a0[idx]]+env_reward[env_idx],
                    ])

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    # Train meta
    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= batch_memory_size:
        meta_train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        batch = []
        num_batch = 0

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    log_episodic_reward.append(np.mean(episode_rew))
    log_length.append(np.mean(steps))
    log_winrate.append(np.mean(envs.blue_win()))

    if log_on:
        tag = 'adapt_train_log/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'reward': log_episodic_reward(),
        }, writer, global_episodes)

    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)
