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
from utility.utils import MovingAverage as MA
from utility.utils import discount_rewards
from utility.buffer import Trajectory
from utility.gae import gae
from utility.multiprocessing import SubprocVecEnv
from utility.logger import record
from utility.RL_Wrapper import TrainedNetwork

from method.ppo import PPO_multimodes as Network


MODE = int(sys.argv[-1])
assert MODE in [0,1,2,3]
num_mode = 3
if MODE == 3:
    ALTERMODE = True
    MODE = 0
else:
    ALTERMODE = False
MODE_NAME = lambda mode: ['_attack', '_scout', '_defense', ''][mode]

map_list = ['fair_map/board{}.txt'.format(i) for i in range(1,4)]
call_map = lambda: random.choice(map_list)

## Training Directory Reset
OVERRIDE = False;
TRAIN_NAME = 'golub_ppo_subpolicies_backup'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
GPU_CAPACITY = 0.90
NENV = multiprocessing.cpu_count()

if OVERRIDE:
    #  Remove and reset log and model directory
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH,ignore_errors=True)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH,ignore_errors=True)

# Create model and log directory
if not os.path.exists(MODEL_PATH):
    try:
        os.makedirs(MODEL_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {MODEL_PATH} failed')
if not os.path.exists(LOG_PATH):
    try:
        os.makedirs(LOG_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {LOG_PATH} failed')

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
save_image_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*16
moving_average_step = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame = config.getint('DEFAULT', 'KEEP_FRAME')
map_size = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 128
epoch = 2
minbatch_size = 8000

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
setting_paths = ['setting_ppo_attacker.ini', 'setting_ppo_scout.ini', 'setting_ppo_defense.ini']
red_policy = policy.Roomba()
if MODE == 2:
    red_policy = policy.AStar()
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size, policy_red=red_policy,
	config_path=setting_paths[MODE])

envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

sess = tf.Session(config=config)
progbar = tf.keras.utils.Progbar(total_episodes,interval=10)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
subtrain_step = [tf.Variable(0, trainable=False) for _ in range(num_mode)]
subtrain_step_next = [tf.assign_add(step, NENV) for step in subtrain_step]
network = Network(in_size=input_size, action_size=action_space, scope='main', sess=sess, num_mode=num_mode, model_path=MODEL_PATH)

def train(trajs, bootstrap=0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
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
            network.update_global(
                state, action, tdtarget, advantage, old_logit, global_episodes, writer, log, MODE)

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3)
network.initiate(saver, MODEL_PATH)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
global_episodes = sess.run(global_step) # Reset the counter
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

# Red Policy (selfplay)
'''
red_policy = TrainedNetwork(
            model_path='model/a3c_pretrained',
            input_tensor='global/state:0',
            output_tensor='global/actor/Softmax:0'
        )
'''


def reward_shape(prev_red_alive, red_alive, done, idx=None):
    prev_red_alive = np.reshape(prev_red_alive, [NENV, num_red])
    red_alive = np.reshape(red_alive, [NENV, num_red])
    reward = []
    red_flags = envs.red_flag()
    blue_flags = envs.blue_flag()
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
            else:
                reward.append(0)
    return np.array(reward)

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

def get_action(states):
    a1, v1, logits1 = network.run_network(states, MODE)
    actions = np.reshape(a1, [NENV, num_blue])
    return a1, v1, logits1, actions

batch = []
num_batch = 0
while True:
    log_on = interv_cntr(global_episodes, save_stat_frequency, 'log')
    log_image_on = interv_cntr(global_episodes, save_image_frequency, 'im_log')
    save_on = interv_cntr(global_episodes, save_network_frequency, 'save')
    
    # Bootstrap
    if ALTERMODE:
        MODE = np.random.randint(3)
        s1 = envs.reset(config_path=setting_paths[MODE])
    else:
        s1 = envs.reset()
    if np.random.random() < 0.5:
        s1 = envs.reset(custom_board=call_map())
    num_blue = len(envs.get_team_blue()[0])
    num_red = len(envs.get_team_red()[0])
    
    # initialize parameters 
    episode_rew = np.zeros(NENV)
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
        reward = reward_shape(was_alive_red, is_alive_red, done, MODE) - 0.01
        env_reward = (raw_reward - prev_rew)/100
        episode_rew += reward

        if step == max_ep:
            env_reward[:] = -1
            done[:] = True
    
        a1, v1, logits1, actions = get_action(s1)
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

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

    global_episodes += NENV
    sess.run(global_step_next)
    sess.run(subtrain_step_next[MODE])
    # progbar.update(global_episodes)

    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])

    if num_batch >= minbatch_size:
        train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        batch = []
        num_batch = 0
        log_on = True

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    global_episode_rewards.append(np.mean(episode_rew))
    global_length.append(np.mean(steps))
    global_succeed.append(np.mean(envs.blue_win()))

    if log_on:
        step = sess.run(subtrain_step[MODE])
        record({
            'Records/mean_length'+MODE_NAME(MODE): global_length(),
            'Records/mean_succeed'+MODE_NAME(MODE): global_succeed(),
            'Records/mean_episode_reward'+MODE_NAME(MODE): global_episode_rewards(),
        }, writer, step)
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

