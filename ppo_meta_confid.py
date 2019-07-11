import pickle
import os
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

from method.ppo import PPO as Network
from method.ppo import PPO_multimodes as SubNetwork

num_mode = 3

## Training Directory Reset
OVERRIDE = False;
TRAIN_NAME = 'meta_confid'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME

SUBP_TRAIN_NAME = 'ppo_subp_robust'
SUBP_LOG_PATH = './logs/'+SUBP_TRAIN_NAME
SUBP_MODEL_PATH = './model/' + SUBP_TRAIN_NAME
GPU_CAPACITY = 0.90

NENV = multiprocessing.cpu_count()  
LOGDEVICE = False
ENV_SETTING_PATH = 'setting_ppo_meta.ini'

## Data Path
path_create(LOG_PATH, override=OVERRIDE)
path_create(MODEL_PATH, override=OVERRIDE)
path_create(SAVE_PATH, override=OVERRIDE)
path_create(SUBP_LOG_PATH, override=OVERRIDE)
path_create(SUBP_MODEL_PATH, override=OVERRIDE)

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
minibatch_size = 128
epoch = 2
batch_memory_size = 4000
batch_meta_memory_size = 4000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
global_episode_rewards = MovingAverage(moving_average_step)
global_length = MovingAverage(moving_average_step)
global_succeed = MovingAverage(moving_average_step)
global_meta_freq = MovingAverage(moving_average_step)
global_episodes = 0

## Map Setting
map_dir = 'fair_map/'
map_list = [map_dir+'board{}.txt'.format(i) for i in range(1,5)]
max_epsilon = 0.55; max_at = 150000
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
heur_policy_list = [policy.Patrol(), policy.Roomba(), policy.Defense(), policy.AStar()]
heur_weight = [1,2,1,1]
heur_weight = np.array(heur_weight) / sum(heur_weight)
def use_this_policy():
    return np.random.choice(heur_policy_list, p=heur_weight)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size, policy_red=use_this_policy(),
	config_path=ENV_SETTING_PATH)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOGDEVICE)

sess = tf.Session(config=config)

global_meta_step = tf.Variable(0, trainable=False, name='global_step')
global_meta_step_next = tf.assign_add(global_meta_step, NENV)
meta_network = Network(in_size=input_size, action_size=num_mode, scope='meta', sess=sess, model_path=MODEL_PATH)
meta_saver = tf.train.Saver(max_to_keep=3, var_list=meta_network.get_vars+[global_meta_step])

# Resotre / Initialize
meta_network.initiate(meta_saver, MODEL_PATH)

meta_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
global_episodes_meta = sess.run(global_meta_step) # Reset the counter

meta_network.save(meta_saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes_meta)

## Prepare Subpolicies
#with tf.device('/gpu:1'):
subp_network = SubNetwork(in_size=input_size, action_size=action_space, sess=sess, num_mode=num_mode, scope='main')
subp_saver = tf.train.Saver(var_list=subp_network.get_vars)
subp_network.initiate(subp_saver, SUBP_MODEL_PATH)

def meta_train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
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
        meta_network.update_global(*mdp_tuple, global_episodes, writer, log)

'''
def train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
    def batch_iter(batch_size, states, actions, logits, tdtargets, advantages):
        size = len(states)
        for _ in range(size // batch_size):
            rand_ids = np.random.randint(0, size, batch_size)
            yield states[rand_ids, :], actions[rand_ids], logits[rand_ids], tdtargets[rand_ids], advantages[rand_ids]

    buffer_s, buffer_a, buffer_tdtarget, buffer_adv, buffer_logit = [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]
    for traj in trajs:
        if len(traj) == 0:
            continue
        observations = traj[0]
        actions = traj[1]
        bootstrap = traj[6][-1]
        mode = traj[5][0]
        
        td_target, advantages = gae(traj[2], traj[3], bootstrap,
                gamma, lambd, normalize=False)

        buffer_s[mode].extend(observations)
        buffer_a[mode].extend(actions)
        buffer_tdtarget[mode].extend(td_target)
        buffer_adv[mode].extend(advantages)
        buffer_logit[mode].extend(logits)

    for mode in range(num_mode):
        buffer_size = len(buffer_s[mode])
        if buffer_size < 10:
            return

        for _ in range(epoch):
            for state, action, old_logit, tdtarget, advantage in  \
                batch_iter(batch_size, np.stack(buffer_s[mode]), np.stack(buffer_a[mode]),
                        np.stack(buffer_logit[mode]), np.stack(buffer_tdtarget[mode]), np.stack(buffer_adv[mode])):
                network.update_global(
                    state, action, tdtarget, advantage, old_logit, global_episodes, writer, log, mode)
'''

# Red Policy (selfplay)
'''
red_policy = TrainedNetwork(
            model_path='model/a3c_pretrained',
            input_tensor='global/state:0',
            output_tensor='global/actor/Softmax:0'
        )
'''

'''
def reward_shape(prev_red_alive, red_alive, done, env_reward):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    reward_list = []
    red_flags = envs.red_flag()
    blue_flags = envs.blue_flag()
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
        else:
            reward[2] = 0
        reward_list.append(reward + env_reward[i])
    return np.array(reward_list)
'''

print('Training Initiated:')
entering_confids = np.ones(NENV*num_blue)
playing_mode = np.zeros(NENV*num_blue, dtype=int)
mode_length = np.ones(NENV*num_blue)
freq_list = []
def get_action(states, initial=False):
    global mode_length, freq_list
    if initial:
        confid_value = np.ones((NENV, num_blue))
        mode_length = np.ones(NENV*num_blue)
        freq_list = []

    # Run meta controller
    action, critic, logits = meta_network.run_network(states)
    prob = sess.run(meta_network.actor,
            feed_dict={meta_network.state_input: states})
    confids = -np.mean(prob * np.log(prob), axis=1) # Entropy
    for i in range(NENV*num_blue):
        confid = confids[i]
        old_confid = entering_confids[i]
        if confid < old_confid: # compare inverse entropy
            entering_confids[i] = confid
            playing_mode[i] = action[i]
            freq_list.append(mode_length[i])
            mode_length[i] = 1
        else:
            mode_length[i] += 1

    # Run subp network
    sub_logit = sess.run(subp_network.actor,
            feed_dict={subp_network.state_input:states})
    actions = []
    for i in range(NENV*num_blue):
        mod = playing_mode[i]
        prob = sub_logit[mod][i]; prob=prob/sum(prob)
        actions.append(np.random.choice(action_space, p=prob))
    actions = np.reshape(actions, [NENV, num_blue])

    return action, critic, logits, actions

meta_batch = []
num_meta_batch = 0
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
    was_done = [False for env in range(NENV)]

    meta_trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]
    
    # Bootstrap
    s1 = envs.reset(
            custom_board=use_this_map(global_episodes, max_at, max_epsilon),
            policy_red=use_this_policy()
        )
    a1, v1, logits1, actions = get_action(s1, initial=True)

    # Rollout
    cumul_reward = np.zeros(NENV)
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        env_reward = (raw_reward - prev_rew - 0.1*step)/100
        episode_rew += env_reward

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True

        #reward = reward_shape(was_alive_red, is_alive_red, done, env_reward) - 0.01
        cumul_reward += env_reward
    
        a1, v1, logits1, actions = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                # Reselect meta
                meta_trajs[idx].append([s0[idx], a[idx], env_reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break

    global_episodes_meta += NENV
    sess.run(global_meta_step_next)

    '''
    # Train sub
    if num_batch >= batch_memory_size:
        train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        batch = []
        num_batch = 0
    '''

    # Train meta
    meta_batch.extend(meta_trajs)
    num_meta_batch += sum([len(traj) for traj in meta_trajs])
    if num_meta_batch >= batch_meta_memory_size:
        meta_train(meta_batch, 0, epoch, minibatch_size, meta_writer, log_image_on, global_episodes_meta)
        meta_batch = []
        num_meta_batch = 0

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in meta_trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    global_episode_rewards.append(np.mean(episode_rew))
    global_length.append(np.mean(steps))
    global_succeed.append(np.mean(envs.blue_win()))
    freq_list.append(mode_length.tolist())
    global_meta_freq.append(np.mean(freq_list))

    if log_on:
        record({
            'Records/mean_length': global_length(),
            'Records/mean_succeed': global_succeed(),
            'Records/mean_episode_reward': global_episode_rewards(),
            'Records/mean_subp_length': global_meta_freq(),
        }, meta_writer, global_episodes_meta)
        
    if save_on:
        meta_network.save(meta_saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes_meta)

    if play_save_on:
        for i in range(NENV):
            with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
                pickle.dump(info[i], handle)
