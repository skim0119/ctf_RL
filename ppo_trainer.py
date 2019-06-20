'''
- Self-play
- Flat
- PPO
- No UAV
'''

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

import policy

from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage
from utility.utils import discount_rewards
from utility.buffer import Trajectory
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.ppo import PPO as Network

## Training Directory Reset
OVERRIDE = False;
TRAIN_NAME = 'ppo_flat_single'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
GPU_CAPACITY = 0.90

BOARD_PATH = 'fair_map'
board_list = [BOARD_PATH+'/board{}.txt'.format(idx) for idx in range(1,4)]

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
save_stat_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*4
save_image_frequency = config.getint('LOG', 'SAVE_STATISTICS_FREQ')*16
moving_average_step = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame = config.getint('DEFAULT', 'KEEP_FRAME')
map_size = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 128
epoch = 4
minimum_replay_size = 4000
selfplay_reload = 2048*4

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization 
global_episode_rewards = MovingAverage(moving_average_step)
global_length = MovingAverage(moving_average_step)
global_succeed = MovingAverage(moving_average_step)
global_episodes = 0

## Environment Initialization
env = gym.make('cap-v0', map_size=map_size, config_path='setting_ppo_flat.ini')
num_blue = len(env.get_team_blue)
num_red = len(env.get_team_red)

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
dummy_ = tf.placeholder(tf.int32)
global_step_next = tf.assign_add(global_step, dummy_)
network = Network(in_size=input_size, action_size=action_space, scope='main', sess=sess)
# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3)
network.initiate(saver, MODEL_PATH)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
# Initiate
global_episodes = sess.run(global_step) # Reset the counter
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

## Red Policy (selfplay)
red_policy = TrainedNetwork(
            model_path=MODEL_PATH,
            input_tensor='main/state:0',
            output_tensor='main/actor/Softmax:0'
        )

def train(trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
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
                state, action, tdtarget, advantage, old_logit, global_episodes, writer, log)

print('Training Initiated:')
blue_stack = []
red_stack = []
def process_state(states, stack, agent):
    oh = one_hot_encoder(states, agent, vision_range)
    stack.append(oh)
    stack.pop(0)
    assert len(stack) == keep_frame
    return np.concatenate(stack, axis=3)

def get_action(blue_state_input, red_state_input):
    a1, v1, logits1 = network.run_network(blue_state_input)
    rprob = red_policy.run_network(red_state_input)
    ra = [np.random.choice(action_space, p=p/sum(p)) for p in rprob]
    actions = np.concatenate([a1, ra])
    return a1, v1, logits1, actions

while global_episodes < total_episodes:
    log_on = global_episodes % save_stat_frequency == 0 and global_episodes != 0
    log_image_on = global_episodes % save_image_frequency == 0 and global_episodes != 0
    save_on = global_episodes % save_network_frequency == 0 and global_episodes != 0
    reload_on = global_episodes % selfplay_reload == 0 and global_episodes != 0
    
    # initialize parameters 

    s1 = env.reset(custom_board=random.choice(board_list))
    episode_rew = 0
    prev_rew = 0
    was_alive = [True] * num_blue

    trajs = [Trajectory(depth=5) for _ in range(num_blue)]

    blue_stack = [one_hot_encoder(s1, env.get_team_blue, vision_range) for _ in range(keep_frame)]
    red_stack = [one_hot_encoder(env.get_obs_red, env.get_team_red, vision_range) for _ in range(keep_frame)]
    
    # Bootstrap
    s1 = process_state(s1, blue_stack, env.get_team_blue)
    red_state = process_state(env.get_obs_red, red_stack, env.get_team_red)
    a1, v1, logits1, actions = get_action(s1, red_state)

    # Rollout
    replay_size = 0
    rollout_size = 0
    while replay_size < minimum_replay_size:
        for step in range(max_ep+1):
            s0 = s1
            a, v0 = a1, v1
            logits = logits1

            s1, raw_reward, done, info = env.step(actions)
            is_alive = [agent.isAlive for agent in env.get_team_blue]
            reward = (raw_reward - prev_rew - 0.1*step)

            if step == max_ep:
                reward = -100
                done = True

            reward /= 100.0
            episode_rew += reward
        
            s1 = process_state(s1, blue_stack, env.get_team_blue)
            red_state = process_state(env.get_obs_red, red_stack, env.get_team_red)
            a1, v1, logits1, actions = get_action(s1, red_state)

            # push to buffer
            for idx, agent in enumerate(env.get_team_blue):
                if was_alive[idx]:
                    trajs[idx].append([s0[idx], a[idx], reward, v0[idx], logits[idx]])

            prev_rew = raw_reward
            was_alive = is_alive

            if done:
                break

        for traj in trajs:
            replay_size += len(traj)

        rollout_size += 1
        global_episode_rewards.append(episode_rew)
        global_length.append(step)
        global_succeed.append(env.blue_win)

    train(trajs, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
    global_episodes += rollout_size
    sess.run(global_step_next, feed_dict={dummy_:rollout_size})

    if log_on:
        record({
            'Records/mean_length': global_length(),
            'Records/mean_succeed': global_succeed(),
            'Records/mean_episode_reward': global_episode_rewards(),
        }, writer, global_episodes)
        print('logged')
        
    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)
        print('saved')

    if reload_on:
        red_policy.reset_network_weight()
        print('reloaded')
