'''
- Self-play
- Flat
- PPO
- No UAV
'''

import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

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

from method.ppo import PPO_V3 as Network

OVERRIDE = False
PROGBAR = True
LOG_DEVICE = False

## Training Directory Reset
TRAIN_NAME = 'PPO_V3_Test'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
GPU_CAPACITY = 0.90

if OVERRIDE:
    MODEL_LOAD_PATH = './model/ppo_flat_robust/' # initialize values
else:
    MODEL_LOAD_PATH = MODEL_PATH

NENV = multiprocessing.cpu_count()  
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'setting_full.ini'

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
save_stat_frequency    = 256 #config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = 256 #config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
pretrain_vae_ep = 0
minibatch_size = 512
epoch = 1  # PPO is on-policy, but 2,3 epoch seems to converge due to constraint
minimum_batch_size = 7000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
log_episodic_reward = MovingAverage(moving_average_step)
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

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
    return lambda: gym.make(
            'cap-v0',
            map_size=map_size,
            config_path=env_setting_path
            )
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
nchannel = envs.observation_space.shape[-1]
print(nchannel)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOG_DEVICE)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
network = Network(in_size=input_size, action_size=action_space, scope='main', sess=sess)

# Resotre / Initialize
global_episodes = 0
saver = tf.train.Saver(max_to_keep=3)
network.initiate(saver, MODEL_LOAD_PATH)
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
        network.update_global(*mdp_tuple, global_episodes, writer, log)

def get_action(states):
    a1, v1, logits1 = network.run_network(states)
    actions = np.reshape(a1, [NENV, num_blue])
    return a1, v1, logits1, actions

#### main ####
# Pretrain VAE
if PROGBAR:
    pretrain_progbar = tf.keras.utils.Progbar(pretrain_vae_ep)
for ep in range(pretrain_vae_ep//NENV):
    if PROGBAR:
        pretrain_progbar.update(ep*NENV)
    
    batch = []
    num_batch = 0
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]
    
    s1 = envs.reset(
            config_path=env_setting_path,
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=policy.Random,
            policy_blue=policy.Random
        )
    done = np.full((NENV,), True)

    # Rollout
    for step in range(max_ep+1):
        s0 = s1
        s1, _, done, _= envs.step()
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]

        if step == max_ep:
            done[:] = True


        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                batch.append(s0[idx])

        was_alive = is_alive
        was_done = done

        if np.all(done):
            break
    batch = np.stack(batch)

    it = batch_sampler(512, 1, batch)
    for minibatch in it:
        feed_dict = {network.state_input: minibatch[0]}
        sess.run(network.vae_updater, feed_dict)

# Rollout-train
while True:
    log_on       = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on      = interval_flag(global_episodes, save_network_frequency, 'save')
    reload_on    = False     # interval_flag(global_episodes,selfplay_reload, 'reload')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]

    batch = []
    num_batch = 0
    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    #if global_episodes > 20000: # Partial transition
    #    env_setting_path = 'setting_partial.ini'
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
        reward = (raw_reward - prev_rew - 0.01)/100.0

        if step == max_ep:
            reward[:] = -1
            done[:] = True

        episode_rew += reward

        a1, v1, logits1, actions = get_action(s1)

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([s0[idx], a[idx], reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
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
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        tag = 'V3/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'reward': log_episodic_reward(),
            tag+'rollout_time': log_looptime(),
            tag+'train_time': log_traintime(),
        }, writer, global_episodes)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)

