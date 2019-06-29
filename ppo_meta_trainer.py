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
import gym_cap.envs.const as CONST
import numpy as np
import random
import math

# the modules that you can use to generate the policy. 
import policy

# Data Processing Module
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards
from utility.buffer import Trajectory
from utility.gae import gae
from utility.multiprocessing import SubprocVecEnv
from utility.logger import record
from utility.RL_Wrapper import TrainedNetwork

from method.ppo import PPO_multimodes as subNetwork
from method.ppo import PPO as metaNetwork

from method.base import initialize_uninitialized_vars as iuv

num_mode = 3
MODE_NAME = ['attack', 'scout', 'defense']

## Training Directory Reset
OVERRIDE = False;
TRAIN_NAME = 'ppo_meta'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SUBP_TRAIN_NAME = 'ppo_subpolicies'
SUBP_LOG_PATH = './logs/'+SUBP_TRAIN_NAME
SUBP_MODEL_PATH = './model/' + SUBP_TRAIN_NAME
GPU_CAPACITY = 0.90

if OVERRIDE:
    #  Remove and reset log and model directory
    if os.path.exists(SUBP_LOG_PATH):
        shutil.rmtree(SUBP_LOG_PATH,ignore_errors=True)
    if os.path.exists(SUBP_MODEL_PATH):
        shutil.rmtree(SUBP_MODEL_PATH,ignore_errors=True)

# Create model and log directory
if not os.path.exists(SUBP_MODEL_PATH):
    try:
        os.makedirs(SUBP_MODEL_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {SUBP_MODEL_PATH} failed')
if not os.path.exists(SUBP_LOG_PATH):
    try:
        os.makedirs(SUBP_LOG_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {SUBP_LOG_PATH} failed')

## Import Shared Training Hyperparameters
config = configparser.ConfigParser()
config.read('config.ini')

# Training
total_episodes = config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep = 300 # config.getint('TRAINING', 'MAX_STEP')

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
save_image_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*16
moving_average_step = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame = config.getint('DEFAULT', 'KEEP_FRAME')
map_size = config.getint('DEFAULT', 'MAP_SIZE')
nenv = config.getint('DEFAULT', 'NUM_ENV')

## PPO Batch Replay Settings
minibatch_size = 128
epoch = 2
meta_delay = 10
batch_memory_size = 8000
batch_meta_memory_size = 4000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
global_episode_rewards = MA(moving_average_step)
global_length = MA(moving_average_step)
global_succeed = MA(moving_average_step)
global_episodes = 0

## Environment Initialization
red_policy = policy.roomba.Roomba()
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size, policy_red=red_policy,
	config_path='setting_ppo_meta.ini')
envs = [make_env(map_size) for i in range(nenv)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

sess = tf.Session(config=config, graph=tf.Graph())
meta_sess = tf.Session(config=config, graph=tf.Graph())
progbar = tf.keras.utils.Progbar(total_episodes,interval=10)

with sess.graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step, nenv)
    subtrain_step = [tf.Variable(0, trainable=False) for _ in range(num_mode)]
    subtrain_step_next = [tf.assign_add(step, nenv) for step in subtrain_step]
    network = subNetwork(in_size=input_size, action_size=action_space, scope='main', sess=sess, num_mode=num_mode, model_path=SUBP_MODEL_PATH)
    saver = tf.train.Saver(max_to_keep=3)
with meta_sess.graph.as_default():
    global_meta_step = tf.Variable(0, trainable=False, name='global_step')
    global_meta_step_next = tf.assign_add(global_meta_step, nenv)
    meta_network = metaNetwork(in_size=input_size, action_size=num_mode, scope='meta', sess=meta_sess, model_path=MODEL_PATH)
    meta_saver = tf.train.Saver(max_to_keep=3)

# Resotre / Initialize
network.initiate(saver, SUBP_MODEL_PATH)
meta_network.initiate(meta_saver, MODEL_PATH)

writer = tf.summary.FileWriter(SUBP_LOG_PATH, sess.graph)
meta_writer = tf.summary.FileWriter(LOG_PATH, meta_sess.graph)
global_episodes = sess.run(global_step) # Reset the counter
global_episodes_meta = meta_sess.run(global_meta_step) # Reset the counter
network.save(saver, SUBP_MODEL_PATH+'/ctf_policy.ckpt', global_episodes)
meta_network.save(meta_saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes_meta)

def meta_train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
    def batch_iter(batch_size, states, actions, logits, tdtargets, advantages):
        size = len(states)
        for _ in range(size // batch_size):
            rand_ids = np.random.randint(0, size, batch_size)
            yield states[rand_ids, :], actions[rand_ids], logits[rand_ids], tdtargets[rand_ids], advantages[rand_ids]

    buffer_s, buffer_a, buffer_tdtarget, buffer_adv, buffer_logit = [], [], [], [], []
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        observations = traj[0]
        actions = traj[1]

        td_target, advantages = gae(traj[2], traj[3], 0,
                gamma, lambd, normalize=False)
        
        buffer_s.extend(observations)
        buffer_a.extend(actions)
        buffer_tdtarget.extend(td_target)
        buffer_adv.extend(advantages)
        buffer_logit.extend(traj[4])

    buffer_size = len(buffer_s)
    if buffer_size < 10:
        return

    for _ in range(epoch):
        for state, action, old_logit, tdtarget, advantage in  \
            batch_iter(batch_size, np.stack(buffer_s), np.stack(buffer_a),
                    np.stack(buffer_logit), np.stack(buffer_tdtarget), np.stack(buffer_adv)):
            meta_network.update_global(
                state, action, tdtarget, advantage, old_logit, global_episodes, writer, log)

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
        print(traj[5])
        input('')
        
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

# Red Policy (selfplay)
'''
red_policy = TrainedNetwork(
            model_path='model/a3c_pretrained',
            input_tensor='global/state:0',
            output_tensor='global/actor/Softmax:0'
        )
'''


def reward_shape(prev_red_alive, red_alive, done, env_reward):
    prev_red_alive = np.reshape(prev_red_alive, [nenv, num_red])
    red_alive = np.reshape(red_alive, [nenv, num_red])
    reward_list = []
    red_flags = envs.red_flag()
    blue_flags = envs.blue_flag()
    for i in range(nenv):
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

print('Training Initiated:')
def interv_cntr(step, freq, name):
    count = step // freq
    if not hasattr(interv_cntr, name):
        setattr(interv_cntr, name, count)
    if getattr(interv_cntr, name) < count:
        setattr(interv_cntr, name, count)
        return True
    else:
        return False

def get_meta_action(states):
    a1, v1, logits1 = meta_network.run_network(states)
    actions = np.reshape(a1, [nenv, num_blue])
    return a1, v1, logits1 

def get_action(states, meta):
    a_list ,v_list, logits_list = [],[],[] 
    for mode in range(3):
        a1, v1, logits1 = network.run_network(states, mode)
        a_list.append(a1)
        v_list.append(v1)
        logits_list.append(logits1)
    action = np.array(a_list)[meta]
    crtiic = np.array(v_list)[meta]
    logits = np.array(logits_list)[meta]

    actions = np.reshape(action, [nenv, num_blue])
    return action, critic, logits, actions

meta_batch = []
num_meta_batch = 0
batch = []
num_batch = 0
while global_episodes < total_episodes:
    log_on = interv_cntr(global_episodes, save_stat_frequency, 'log')
    log_image_on = interv_cntr(global_episodes, save_image_frequency, 'im_log')
    save_on = interv_cntr(global_episodes, save_network_frequency, 'save')
    
    # initialize parameters 
    episode_rew = np.zeros(nenv)
    prev_rew = np.zeros(nenv)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(nenv)]

    trajs = [Trajectory(depth=7) for _ in range(num_blue*nenv)]
    meta_trajs = [Trajectory(depth=5) for _ in range(num_blue*nenv)]
    
    # Bootstrap
    s1 = envs.reset()
    meta_a1, meta_v1, meta_logits1 = get_meta_action(s1)
    a1, v1, logits1, actions = get_action(s1, meta_a1)

    # Rollout
    cumul_reward = np.zeros(nenv)
    for step in range(max_ep+1):
        s0 = s1
        meta_a, meta_v0 = meta_a1, meta_v1
        a, v0 = a1, v1
        meta_logits = meta_logits1
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        is_alive_red = [agent.isAlive for agent in envs.get_team_red().flat]
        env_reward = (raw_reward - prev_rew - 0.1*step)/100
        episode_rew += env_reward

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True
        reward = reward_shape(was_alive_red, is_alive_red, done, env_reward) - 0.01
        cumul_reward += env_reward
    
        meta_a1, meta_v1, meta_logits1 = get_meta_action(s1)
        a1, v1, logits1, actions = get_action(s1, meta_a1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([
                    s0[idx],
                    a[idx],
                    reward[env_idx][meta_a[idx]],
                    v0[idx],
                    logits[idx],
                    meta_a[idx],
                    v1[idx]])

        # Reselect meta
        if step % meta_delay == 1:
            for idx, agent in enumerate(envs.get_team_blue().flat):
                env_idx = idx // num_blue
                if was_alive[idx] and not was_done[env_idx]:
                    meta_trajs[idx].append([s0[idx], meta_a[idx], cumul_reward[env_idx], meta_v0[idx], meta_logits[idx]])
            cumul_reward[:] = 0

            batch.extend(trajs)
            num_batch += sum([len(traj) for traj in trajs])
            trajs = [Trajectory(depth=7) for _ in range(num_blue*nenv)]

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break

    global_episodes += nenv
    sess.run(global_step_next)
    sess.run(subtrain_step_next)
    global_episodes_meta += nenv
    meta_sess.run(global_meta_step_next)


    # Train sub
    if num_batch >= batch_memory_size:
        train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        batch = []
        num_batch = 0

    # Train meta
    meta_batch.extend(meta_trajs)
    num_meta_batch += sum([len(traj) for traj in meta_trajs])
    if num_meta_batch >= batch_meta_memory_size:
        meta_train(meta_batch, 0, epoch, minibatch_size, meta_writer, log_image_on, global_episodes_meta)
        meta_batch = []
        num_meta_batch = 0

    steps = []
    for env_id in range(nenv):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    global_episode_rewards.append(np.mean(episode_rew))
    global_length.append(np.mean(steps))
    global_succeed.append(np.mean(envs.blue_win()))


    if log_on:
        record({
            'Records/mean_length': global_length(),
            'Records/mean_succeed': global_succeed(),
            'Records/mean_episode_reward': global_episode_rewards(),
        }, meta_writer, global_episodes_meta)
        
    if save_on:
        network.save(saver, SUBP_MODEL_PATH+'/ctf_policy.ckpt', global_episodes)
        meta_network.save(meta_saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes_meta)

