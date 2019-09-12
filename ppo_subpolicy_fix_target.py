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

assert len(sys.argv) == 7
target_setting_path = sys.argv[6]
device_t = sys.argv[2]

N_ATT = int(sys.argv[3])
N_SCT = int(sys.argv[4])
N_DEF = int(sys.argv[5])

PROGBAR = False
LOGDEVICE = False
RBETA = 0.5

num_mode = 3

## Training Directory Reset
TRAIN_NAME = sys.argv[1]
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.95

NENV = 20 # multiprocessing.cpu_count() 
SWITCH_EP = 0


MODEL_LOAD_PATH = './model/fix_baseline_team{}{}{}/'.format(N_ATT, N_SCT, N_DEF) # initialize values
print(MODEL_LOAD_PATH)
assert os.path.exists(MODEL_LOAD_PATH)
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
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 3
minbatch_size = 2048

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 7 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
log_episodic_reward = MA(moving_average_step)
log_length = MA(moving_average_step)
log_winrate = MA(moving_average_step)

log_attack_reward = MovingAverage(moving_average_step)
log_scout_reward = MovingAverage(moving_average_step)
log_defense_reward = MovingAverage(moving_average_step)

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
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOGDEVICE, allow_soft_placement=True)

sess = tf.Session(config=config)

global_episodes = 0
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
with tf.device(device_t):
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

global_episodes = sess.run(global_step)

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

def reward_shape(prev_red_alive, red_alive, done):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    r = []
    red_flags = envs.red_flag_captured()
    blue_flags = envs.blue_flag_captured()
    for i in range(NENV):
        # Attack (C/max enemy)
        num_prev_enemy = sum(prev_red_alive[i])
        num_enemy = sum(red_alive[i])
        for _ in range(N_ATT):
            r.append((num_prev_enemy - num_enemy)*0.25)
        # Scout
        for _ in range(N_SCT):
            if red_flags[i]:
                r.append(1)
            else:
                r.append(0)
        # Defense
        for _ in range(N_DEF):
            if blue_flags[i]:
                r.append(-1)
            elif done[i]:
                r.append(1)
            else:
                r.append(0)
    return np.array(r)

def ghost_reward(prev_red_alive, red_alive, done):
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


print('Training Initiated:')
def get_action(states):
    cnt1 = N_ATT
    cnt2 = cnt1 + N_SCT
    a1, v1, logits1 = [], [], []
    res = network.run_network_all(states)
    a, v, logits = res[:3]
    a1.extend(a[:cnt1]); v1.extend(v[:cnt1]); logits1.extend(logits[:cnt1])
    a, v, logits = res[3:6]
    a1.extend(a[cnt1:cnt2]); v1.extend(v[cnt1:cnt2]); logits1.extend(logits[cnt1:cnt2])
    a, v, logits = res[6:]
    a1.extend(a[cnt2:]); v1.extend(v[cnt2:]); logits1.extend(logits[cnt2:])

    actions = np.reshape(a1, [NENV, num_blue])
    return np.array(a1), np.array(v1), np.array(logits1), actions

batch_att = []
batch_sct = []
batch_def = []
num_batch_att = 0
num_batch_sct = 0
num_batch_def = 0
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')
    
    # initialize parameters 
    if global_episodes > SWITCH_EP:
        env_setting_path = target_setting_path
    s1 = envs.reset(
            config_path=ENV_SETTING_PATH,
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=use_this_policy()
        )
    num_blue = len(envs.get_team_blue()[0])
    num_red = len(envs.get_team_red()[0])

    episode_rew = np.zeros(NENV)
    case_rew = [np.zeros(NENV) for _ in range(3)]
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    a1, v1, logits1, actions = get_action(s1)

    # Rollout
    stime = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1[:], v1[:]
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        env_reward = (raw_reward-prev_rew-0.01)/100.0

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True

        reward = reward_shape(was_alive_red, is_alive_red, done)
        episode_rew += env_reward

        shaped_reward = ghost_reward(was_alive_red, is_alive_red, done)
        for i in range(NENV): 
            if not was_done[i]:
                for j in range(3):
                    case_rew[j][i] += shaped_reward[i,j]
    
        a1, v1, logits1, actions = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                reward_function = (RBETA) * reward[idx] + (1-RBETA) * env_reward[env_idx]
                trajs[idx].append([s0[idx], a[idx], reward_function, v0[idx], logits[idx]])

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
    
    for i in range(NENV):
        j = 0
        for _ in range(N_ATT):
            batch_att.append(trajs[4*i+j])
            num_batch_att += len(trajs[4*i+j])
            j += 1
        for _ in range(N_SCT): 
            batch_sct.append(trajs[4*i+j])
            num_batch_sct += len(trajs[4*i+j])
            j += 1
        for _ in range(N_DEF): 
            batch_def.append(trajs[4*i+j])
            num_batch_def += len(trajs[4*i+j])
            j += 1

    if num_batch_att >= minbatch_size or num_batch_sct >= minbatch_size or num_batch_def >= minbatch_size::
        stime = time.time()
        train(batch_att, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=0)
        batch_att = []
        num_batch_att = 0

        train(batch_sct, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=1)
        batch_sct = []
        num_batch_sct = 0

        train(batch_def, 0, epoch, minibatch_size, writer, log_image_on, global_episodes, mode=2)
        batch_def = []
        num_batch_def = 0

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())

    log_attack_reward.extend(case_rew[0].tolist())
    log_scout_reward.extend(case_rew[1].tolist())
    log_defense_reward.extend(case_rew[2].tolist())

    if log_on:
        step = sess.run(global_step)
        tag = 'kerasTest/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'reward': log_episodic_reward(),
            tag+'reward_attack': log_attack_reward(),
            tag+'reward_scout': log_scout_reward(),
            tag+'reward_defense': log_defense_reward(),
        }, writer, step)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)
