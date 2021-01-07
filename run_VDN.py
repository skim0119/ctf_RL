import argparse
import configparser
import multiprocessing
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import shutil
import signal
import sys
import threading

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import math
import random
import time
from collections import defaultdict

import gym
import gym_cap
import gym_cap.heuristic as policy
import numpy as np

from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.gae import gae
from utility.logger import *
from utility.multiprocessing import SubprocVecEnv
from utility.utils import MovingAverage, interval_flag, path_create

from method.VDN import VDN_Module as Network

parser = argparse.ArgumentParser(description="PPO trainer for convoy")
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
TRAIN_NAME = "VDN_{}_{:02d}_convoy_{}g{}a_{}g{}a_m{}".format(
    args.machine,
    args.train_number,
    args.nbg,
    args.nba,
    args.nrg,
    args.nra,
    args.map_size,
)
TRAIN_TAG = "IV e2e model w Stacked Frames: " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
MAP_PATH = "./fair_3g_20"
GPU_CAPACITY = 0.95

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
# Batch Replay Settings
minibatch_size = 256
epoch = 1
minimum_batch_size = 2048

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
# map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH) if path[:5]=='board']
def make_env(map_size):
    return lambda: gym.make("cap-v0", map_size=map_size, config_path=game_config)


envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame=keep_frame, size=vision_dx)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])
num_agent = num_blue  # + num_red

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
network = Network(
    input_shape=input_size,
    action_size=action_space,
    agent_type=agent_type,
    save_path=MODEL_PATH,
)

# Resotre / Initialize
global_episodes = network.initiate()
print(global_episodes)

writer = tf.summary.create_file_writer(LOG_PATH)
# network.save(global_episodes)

### TRAINING ###
def train(
    network,
    trajs_critic,
    bootstrap=0.0,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    datasets_critic = []

    # Prepare critic loss dataset
    traj_buffer = defaultdict(list)
    for traj in trajs_critic:
        reward = traj[1]
        critic = traj[2]
        _critic = traj[3][-1]

        td_target, _ = gae(
            reward, critic, _critic,
            gamma, lambd, normalize=False
        )

        traj_buffer["state"].extend(traj[0])
        traj_buffer["td_target"].extend(td_target)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]).astype(np.float32),
                "td_target": np.stack(traj_buffer["td_target"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .repeat(epoch)
        .batch(batch_size)
    )
    datasets_critic = train_dataset

    network.update_network(
        datasets_critic, agent_type_index, writer=writer, step=step, tag="losses/", log=log
    )
    if log:
        with writer.as_default():
            tag = "debug/"
            tb_log_histogram(
                np.array(advantage_list), tag + "dec_advantages", step=step
            )
            writer.flush()


def get_action(states):
    # State Process
    states_list = []
    for mask in agent_type_masking:
        state = np.compress(mask, states, axis=0)
        states_list.append(state)

    # Run network
    results = network.run_network(states_list)

    # Container
    a1 = np.empty([NENV * num_blue], dtype=np.int32)
    v1 = np.empty([NENV * num_blue], dtype=np.float32)

    # Postprocessing
    for (a, v), mask in zip(results, agent_type_masking):
        a1[mask] = a
        v1[mask] = v
    actions = np.reshape(a1, [NENV, num_blue])
    return actions, a1, v1


batch2 = []
num_batch = 0
while global_episodes < total_episodes:

    # initialize parameters
    episode_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]

    trajs_team = [Trajectory(depth=4) for _ in range(NENV)]

    # Bootstrap
    s1 = envs.reset(
        map_size=map_size,
        # map_pool=map_list,
        config_path=game_config,
        policy_red=policy.Roomba,
    )
    s1 = s1.astype(np.float32)
    actions, a1, v1 = get_action(s1)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep):
        s0 = s1
        a0, v0 = a1, v1

        s1, reward, done, info = envs.step(actions)
        s1 = s1.astype(np.float32)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        episode_rew += reward

        actions, a1, v1 = get_action(s1)

        # push to buffer
        v0s = v0.reshape([NENV, num_agent]).sum(axis=1)
        v1s = v1.reshape([NENV, num_agent]).sum(axis=1)
        for env_idx in range(NENV):
            trajs_team[env_idx].append([
                s0[env_idx*num_agent:(env_idx+1)*num_agent],
                reward[env_idx],
                v0s[env_idx],
                v1s[env_idx],
            ])

        was_alive = is_alive
        was_done = done
    etime_roll = time.time()

    batch2.extend(trajs_team)
    num_batch = len(batch2) * 200 * num_agent
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, "im_log")
        train(
            network,
            batch2,
            0,
            epoch,
            minibatch_size,
            writer,
            log_image_on,
            global_episodes,
        )
        etime_train = time.time()
        batch2 = []
        num_batch = 0
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
            tf.summary.scalar(
                tag + "redwin-rate", log_redwinrate(), step=global_episodes
            )
            tf.summary.scalar(
                tag + "env_reward", log_episodic_reward(), step=global_episodes
            )
            tf.summary.scalar(
                tag + "rollout_time", log_looptime(), step=global_episodes
            )
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_episodes)
            writer.flush()

    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)