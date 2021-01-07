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

from method.ActorCritic import PPO_Module_SC2 as Network
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.gae import gae
from utility.logger import *
from utility.multiprocessing import SubprocVecEnv
from utility.utils import MovingAverage, interval_flag, path_create

from smac.env import StarCraft2Env

parser = argparse.ArgumentParser(description="PPO trainer for convoy")
parser.add_argument("--train_number", type=int, help="training train_number",default = 1)
parser.add_argument("--machine", type=str, help="training machine",default = "Final")
parser.add_argument("--map", type=str, help="map name", default = "8m", required=False)
parser.add_argument(
    "--silence", action="store_false", help="call to disable the progress bar"
)
args = parser.parse_args()

PROGBAR = args.silence

## Training Directory Reset
TRAIN_NAME = "PPO_{}_{:02d}_map_{}".format(
    args.machine,
    args.train_number,
    args.map,
)
TRAIN_TAG = "PPO e2e model w Stacked Frames: " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
GPU_CAPACITY = 0.95

NENV = 1
print("Number of cpu_count : {}".format(NENV))

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)

## Import Shared Training Hyperparameters
# Training
total_episodes = 250000
max_ep = 200
gamma = 0.98  # GAE - discount
lambd = 0.98  # GAE - lambda
# Log
save_network_frequency = 1024
save_stat_frequency = 128
save_image_frequency = 128
moving_average_step = 256  # MA for recording episode statistics
# Environment/Policy Settings
keep_frame = 1
# Batch Replay Settings
minibatch_size = 128
epoch = 2
minimum_batch_size = 4096

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
# map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH) if path[:5]=='board']
def make_env():
    return lambda: StarCraft2Env(args.map)

envs = StarCraft2Env(args.map)
from SC2Wrappers import SMACWrapper
envs = SMACWrapper(envs)
envs.reset()
action_space = len(envs.env.get_avail_agent_actions(0))
input_size = [None,envs.env.get_obs_size()]

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Agent Type Setup
agent_type_sc = []
for i in range(len(envs.env.agents)):
    agent_type_sc.append(envs.env.agents[i].unit_type)
agent_type_list = agent_type_sc.copy()
agent_type = [0]*len(set(agent_type_list))
for i,val in enumerate(set(agent_type_sc)):
    agent_type_list = [i if val==agent_t else agent_t for agent_t in agent_type_list ]
for val in agent_type_list:
    agent_type[val] += 1

num_agent = sum(agent_type)
# agent_type = [8]
num_type = len(agent_type)
agent_type_masking = np.zeros([num_type, num_agent], dtype=bool)
agent_type_index = np.zeros([num_agent], dtype=int)
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
    trajs,
    bootstrap=0.0,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    train_datasets = []

    advantage_list = []
    traj_buffer_list = [defaultdict(list) for _ in range(num_type)]
    for trajs in trajs:
        for idx, traj in enumerate(trajs):
            atype = agent_type_index[idx]

            reward = traj[2]
            critic = traj[3]
            _critic = traj[5][-1]

            td_target, advantages = gae(
                reward, critic, _critic,
                gamma, lambd, normalize=False
            )
            advantage_list.append(advantages)

            traj_buffer = traj_buffer_list[atype]
            traj_buffer["state"].extend(traj[0])
            traj_buffer["action"].extend(traj[1])
            traj_buffer["td_target"].extend(td_target)
            traj_buffer["advantage"].extend(advantages)
            traj_buffer["old_log_logit"].extend(traj[4])
            traj_buffer["old_value"].extend(traj[3])

    for atype in range(num_type):
        traj_buffer = traj_buffer_list[atype]
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer["state"]).astype(np.float32),
                    "old_log_logit": np.stack(traj_buffer["old_log_logit"]).astype(np.float32),
                    "action": np.stack(traj_buffer["action"]),
                    "advantage": np.stack(traj_buffer["advantage"]).astype(np.float32),
                    "td_target": np.stack(traj_buffer["td_target"]).astype(np.float32),
                    "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
                }
            )
            .shuffle(64)
            .repeat(epoch)
            .batch(batch_size)
        )
        train_datasets.append(train_dataset)

    network.update_network(
        train_datasets, writer=writer, step=step, tag="losses/", log=log
    )
    if log:
        with writer.as_default():
            tag = "debug/"
            tb_log_histogram(
                np.array(advantage_list), tag + "dec_advantages", step=step
            )
            writer.flush()


def get_action(states,validActions):
    # State Process
    states_list = []
    validActions_list = []
    for mask in agent_type_masking:
        state = np.compress(mask, states, axis=0)
        states_list.append(state)
        validAction = np.compress(mask, validActions, axis=0)
        validActions_list.append(validAction)

    # Run network
    results = network.run_network(states_list,validActions_list)

    # Container
    a1 = np.empty([NENV * num_agent], dtype=np.int32)
    v1 = np.empty([NENV * num_agent], dtype=np.float32)
    p1 = np.empty([NENV * num_agent, action_space], dtype=np.float32)

    # Postprocessing
    for (a, v, p), mask in zip(results, agent_type_masking):
        a1[mask] = a
        v1[mask] = v
        p1[mask, :] = p
    actions = np.reshape(a1, [num_agent])
    return actions, a1, v1, p1


batch = []
num_batch = 0
#while global_episodes < total_episodes:
while True:

    # initialize parameters
    episode_rew = np.zeros(NENV)

    #trajs = [Trajectory(depth=6) for _ in range(num_blue * NENV)]
    trajs = [[Trajectory(depth=6) for _ in range(num_agent)] for _ in range(NENV)]

    # Bootstrap
    s1 = envs.reset()
    s1 = np.stack(s1).astype(np.float32)
    validActions = envs.get_avail_actions()
    validActions = envs.get_avail_actions()

    actions, a1, v1, p1 = get_action(s1,validActions)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep):
        s0 = s1
        a0, v0 = a1, v1
        p0 = p1

        s1, reward, done, info = envs.step(actions)
        s1 = np.stack(s1).astype(np.float32)
        is_alive = [not va[0] for va in envs.get_avail_actions()]
        episode_rew += reward

        validActions = envs.get_avail_actions()
        actions, a1, v1, p1 = get_action(s1,validActions)

        # push to buffer
        for env_idx in range(NENV):
            for agent_id in range(num_agent):
                idx = env_idx * num_agent + agent_id
                trajs[env_idx][agent_id].append(
                    [s0[idx], a0[idx], reward, v0[idx], p0[idx], v1[idx]]
                )

        was_alive = is_alive
        was_done = done

        #if np.all(done):
        #    break
    etime_roll = time.time()

    batch.extend(trajs)
    num_batch = len(batch) * 200 * num_agent
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, "im_log")
        train(
            network,
            batch,
            0,
            epoch,
            minibatch_size,
            writer,
            log_image_on,
            global_episodes,
        )
        etime_train = time.time()
        batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)

    log_episodic_reward.extend(episode_rew.tolist())
    log_winrate.append(info["battle_won"])
    # log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    if PROGBAR:
        progbar.update(global_episodes)

    log_on = interval_flag(global_episodes, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "win-rate", log_winrate(), step=global_episodes)
            # tf.summary.scalar(
            #     tag + "redwin-rate", log_redwinrate(), step=global_episodes
            # )
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
