import pickle

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import shutil
import argparse
import configparser

import signal
import threading
import multiprocessing

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import time
import gym
import gym_cap
import numpy as np
import random
import math
from collections import defaultdict
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import gym_cap.heuristic as policy

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.logger import *
from utility.gae import gae

# from utility.slack import SlackAssist

from method.COMA import COMA as Network

parser = argparse.ArgumentParser(description="CVDC(learnability) trainer for convoy")
parser.add_argument("--train_number", type=int, help="training train_number")
parser.add_argument("--machine", type=str, help="training machine")
parser.add_argument("--map_size", type=int, help="map size")
parser.add_argument("--nbg", type=int, help="number of blue ground")
parser.add_argument("--nba", type=int, help="number of blue air")
parser.add_argument("--nrg", type=int, help="number of red air")
parser.add_argument("--nra", type=int, help="number of red air")
parser.add_argument(
    "--silence", action="store_false", help="call to disable the progress bar"
)
args = parser.parse_args()

PROGBAR = args.silence

## Training Directory Reset
TRAIN_NAME = "COMA_{}_{:02d}_convoy_{}g{}a_{}g{}a_m{}".format(
    args.machine,
    args.train_number,
    args.nbg,
    args.nba,
    args.nrg,
    args.nra,
    args.map_size,
)
TRAIN_TAG = "Central value decentralized control(COMA), " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
MAP_PATH = "./fair_3g_20"
GPU_CAPACITY = 0.95

# slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")

NENV = multiprocessing.cpu_count() // 4
print("Number of cpu_count : {}".format(NENV))

env_setting_path = "env_setting_convoy.ini"
game_config = configparser.ConfigParser()
game_config.read(env_setting_path)
game_config["elements"]["NUM_BLUE"] = str(args.nbg)
game_config["elements"]["NUM_BLUE_UAV"] = str(args.nba)
game_config["elements"]["NUM_RED"] = str(args.nrg)
game_config["elements"]["NUM_RED_UAV"] = str(args.nra)

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)

## Import Shared Training Hyperparameters
config_path = "config.ini"
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = 100000
max_ep = 200
gamma = 0.98  # GAE - discount
lambd = 0.98  # GAE - lambda
# Log
save_network_frequency = 1024
save_stat_frequency = 128
save_image_frequency = 128
moving_average_step = 256  # MA for recording episode statistics
# Environment/Policy Settings
action_space = 5
keep_frame = 4
map_size = args.map_size
vision_range = map_size - 1
vision_dx, vision_dy = 2 * vision_range + 1, 2 * vision_range + 1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]
cent_input_size = [None, map_size, map_size, nchannel]
## Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 1024

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
# map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH) if path[:5]=='board']
_qenv = gym.make("cap-v0", map_size=map_size, config_path=game_config)


def make_env(map_size):
    return lambda: gym.make("cap-v0", map_size=map_size, config_path=game_config)


envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame=keep_frame, size=vision_dx)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])
num_agent = num_blue  # +num_red

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Agent Type Setup
agent_type = []
if args.nba != 0:
    agent_type.append(args.nba)
if args.nbg != 0:
    agent_type.append(args.nbg)
num_type = len(agent_type)
agent_type_masking = np.zeros([num_type, num_blue], dtype=bool)
agent_type_index = np.zeros([num_blue], dtype=int)
prev_i = 0
for idx, i in enumerate(np.cumsum(agent_type)):
    agent_type_masking[idx, prev_i:i] = True
    agent_type_index[prev_i:i] = idx
    prev_i = i
agent_type_masking = np.tile(agent_type_masking, NENV)

# Network Setup
atoms = 128
network = Network(
    state_shape=input_size,
    action_size=action_space,
    num_agent=num_agent,
    agent_type=agent_type,
    atoms=atoms,
    save_path=MODEL_PATH,
)

# Resotre / Initialize
global_episodes = network.initiate()
print(global_episodes)
# input('start?')

writer = tf.summary.create_file_writer(LOG_PATH)

### TRAINING ###
def train_decentral(
    agent_trajs,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):

    # Agent trajectory processing
    traj_buffer_list = [defaultdict(list) for _ in range(num_type)]
    traj_buffer_central = defaultdict(list)
    for traj_i,traj in enumerate(agent_trajs):
        meta_state = np.array(traj[0], dtype=np.float32) # [#, num_agent, lx, ly, ch]
        actions = np.array(traj[1], dtype=int) # [#, num_agent]
        reward = np.array(traj[2], dtype=np.float32) # [#]
        pi = np.array(traj[3], dtype=np.float32) # [#, num_agent, num_action]
        length = meta_state.shape[0]

        meta_action = np.zeros(length, dtype=int)
        #meta_pi = np.ones(meta_state.shape[0], dtype=np.float32)
        for i in range(num_agent):
            meta_action += actions[:,i]*(5**i)
            #meta_pi *= np.take(pi[:,i], actions[:,i], axis=1)
        Q, Q_s_a = network.run_network_central(meta_state, meta_action)
        Q = Q.numpy()
        Q_s_a = Q_s_a.numpy()
        Q_ext = np.concatenate([Q, np.zeros_like(Q[0:1,:])], axis=0)

        meta_pi = None
        for i in range(num_agent):
            new_shape = [length] + [1] * num_agent
            new_shape[1+i] = action_space
            pi_i = pi[:,i].reshape(new_shape)
            if meta_pi is None:
                meta_pi = pi_i
            else:
                meta_pi = pi_i * meta_pi
        meta_pi = meta_pi.reshape([length, -1])

        # Central : Compute central difference
        td_target = reward + gamma * np.sum(meta_pi * Q_ext[1:], axis=1)

        traj_buffer_central["metastate"].extend(list(meta_state))
        traj_buffer_central["metaaction"].extend(list(meta_action))
        traj_buffer_central["td_target"].extend(list(td_target))
        '''
        if traj_i == 0:
            traj_buffer_central["metastate"] = meta_state
            traj_buffer_central["metaaction"] = meta_action
            traj_buffer_central["td_target"] = td_target
        else:
            traj_buffer_central["metastate"] = np.concatenate((traj_buffer_central["metastate"],meta_state))
            traj_buffer_central["metaaction"] = np.concatenate((traj_buffer_central["metaaction"],meta_action))
            traj_buffer_central["td_target"] = np.concatenate((traj_buffer_central["td_target"],td_target))
        '''

        # Decentral : Compute advantage V
        for idx in range(num_agent): # Target specific agent
            indiv_Q = np.zeros([length, action_space])
            for fix_act in range(action_space):
                meta_action_fix_agent = np.zeros(length, dtype=int)
                for i in range(num_agent):
                    if i == idx:
                        meta_action_fix_agent += fix_act*(5**i)
                    else:
                        meta_action_fix_agent += actions[:,i]*(5**i)
                for e in range(length):
                    indiv_Q[e,fix_act] = Q[e,meta_action_fix_agent[e]]
                #indiv_Q[:,fix_act] = np.take(Q, meta_action_fix_agent, axis=1)
            indiv_adv = Q_s_a - np.sum(indiv_Q * pi[:,idx,:], axis=1)

            atype = agent_type_index[idx]
            traj_buffer = traj_buffer_list[atype]
            traj_buffer["state"].extend(list(meta_state[:,idx,...]))
            traj_buffer["action"].extend(list(actions[:,idx]))
            traj_buffer["advantage"].extend(list(indiv_adv))
            '''
            if traj_i == 0:
                traj_buffer["state"] = meta_state[:,idx,...]
                traj_buffer["action"] = actions[:,idx]
                traj_buffer["advantage"] = indiv_adv
            else:
                traj_buffer["state"] = np.concatenate((traj_buffer["state"],meta_state[:,idx,...]))
                traj_buffer["action"] = np.concatenate((traj_buffer["action"], actions[:,idx]))
                traj_buffer["advantage"] = np.concatenate((traj_buffer["advantage"],indiv_adv))
            '''

    dataset_decentral = []
    for atype in range(num_type):
        traj_buffer = traj_buffer_list[atype]
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer["state"]).astype(np.float32),
                    "action": np.stack(traj_buffer["action"]).astype(int),
                    "advantage": np.stack(traj_buffer["advantage"]).astype(np.float32),
                }
            )
            .shuffle(64)
            .repeat(epoch)
            .batch(batch_size)
        )
        dataset_decentral.append(train_dataset)
    dataset_central = (
        tf.data.Dataset.from_tensor_slices(
            {
                "metastate": np.stack(traj_buffer_central["metastate"]).astype(np.float32),
                "metaaction": np.stack(traj_buffer_central["metaaction"]).astype(int),
                "td_target": np.stack(traj_buffer_central["td_target"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .repeat(epoch)
        .batch(batch_size)
    )

    network.update(
        dataset_decentral, dataset_central, writer=writer, log=log, step=step, tag="losses/"
    )
    '''
    if log:
        with writer.as_default():
            writer.flush()
    '''

def run_network(states):
    states_list = []
    for mask in agent_type_masking:
        state = np.compress(mask, states, axis=0)
        states_list.append(state)

    # Run network
    results = network.run_network_decentral(states_list)

    # Container
    a1 = np.empty([NENV * num_agent], dtype=np.int32)
    logits1 = np.empty([NENV * num_agent, action_space], dtype=np.float32)

    # Postprocessing
    for actor,  mask in zip(results, agent_type_masking):
        a = actor['action'].numpy().ravel()
        logits = actor["softmax"].numpy()

        a1[mask] = a
        logits1[mask, :] = logits
    action = np.reshape(a1, [NENV, num_blue])
    return a1, action, logits1


batch = []
dec_batch = []
num_batch = 0
dec_batch_size = 0
#while global_episodes < total_episodes:
while True:
    # initialize parameters
    episode_rew = np.zeros(NENV)
    is_alive = [True for agent in envs.get_team_blue().flat]
    is_done = [False for env in range(NENV * num_agent)]

    trajs = [Trajectory(depth=5) for _ in range(NENV)]

    # Bootstrap
    s1 = envs.reset(config_path=game_config, policy_red=policy.Roomba,)
    s1 = s1.astype(np.float32)

    a1, action, logits1 = run_network(s1)

    # Rollout
    stime_roll = time.time()

    for step in range(max_ep):
        s0 = s1
        a0 = a1
        logits0 = logits1
        was_alive = is_alive
        was_done = is_done

        # Run environment
        s1, reward, done, history = envs.step(action)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        s1 = s1.astype(np.float32)  # Decentralize observation
        episode_rew += reward

        # Run decentral network
        a1, action, logits1 = run_network(s1)

        # Buffer
        for env_idx in range(NENV):
            idx = env_idx * num_agent
            trajs[env_idx].append(
                [
                    s0[idx:idx+num_agent],
                    a0[idx:idx+num_agent],
                    reward[env_idx], 
                    logits0[idx:idx+num_agent],
                ]
            )

    etime_roll = time.time()

    # decentralize training
    dec_batch.extend(trajs)
    dec_batch_size = len(dec_batch) * 200 * num_agent
    if dec_batch_size > minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, "im_log")
        train_decentral(
            dec_batch,
            epoch=epoch,
            batch_size=minibatch_size,
            writer=writer,
            log=log_image_on,
            step=global_episodes,
        )
        etime_train = time.time()
        dec_batch = []
        dec_batch_size = 0
        log_traintime.append(etime_train - stime_train)

    log_episodic_reward.extend(episode_rew.tolist())
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    if PROGBAR:
        progbar.update(global_episodes)

    log_on = interval_flag(global_episodes, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "win-rate", log_winrate(), step=global_episodes)
            tf.summary.scalar(tag + "redwin-rate", log_redwinrate(), step=global_episodes)
            tf.summary.scalar(tag + "env_reward", log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag + "rollout_time", log_looptime(), step=global_episodes)
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_episodes)
            writer.flush()

    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)
