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
#tf.config.experimental_run_functions_eagerly(True)
physical_devices = tf.config.experimental.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import time
import gym
import numpy as np
import random
import math
from collections import defaultdict
from functools import partial

from smac.env import StarCraft2Env

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.logger import *
from utility.gae import gae

# from utility.slack import SlackAssist

from method.CVDC import SF_CVDC as Network

parser = argparse.ArgumentParser(description="CVDC(learnability) trainer for convoy")
parser.add_argument("--train_number", type=int, help="training train_number")
parser.add_argument("--machine", type=str, help="training machine")
parser.add_argument("--map", type=str, default='8m', help='the map of the game')
parser.add_argument("--silence", action="store_false", help="call to disable the progress bar")
parser.add_argument("--print", action="store_true", help="print out the progress in detail")
parser.add_argument("--difficulty", type=str, default='7', help='the difficulty of the game')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--step_mul", type=int, default=8, help='how many steps to make an action')
parser.add_argument("--training_episodes", type=int, default=1000000, help='number of training episodes')
args = parser.parse_args()

PROGBAR = args.silence
PRINT = args.print

## Training Directory Reset
TRAIN_NAME = "CVDC_{}_{}_{:02d}".format(
    args.machine,
    args.map,
    args.train_number,
)
#slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")
TRAIN_TAG = "Central value decentralized control(learnability), " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
SAVE_PATH = "./save/" + TRAIN_NAME
GPU_CAPACITY = 0.95

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

# Training
total_episodes = args.training_episodes
gamma = 0.98  # GAE - discount
lambd = 0.98  # GAE - lambda

# Log
save_network_frequency = 1024
save_stat_frequency = 128
save_image_frequency = 128
moving_average_step = 256  # MA for recording episode statistics

## Environment
env = StarCraft2Env(
    map_name=args.map,
    step_mul=args.step_mul,
    difficulty=args.difficulty,
    replay_dir=SAVE_PATH
)
env_info = env.get_env_info()
print(env_info)

# Environment/Policy Settings
action_space = env_info["n_actions"]
num_agent = env_info["n_agents"]
state_shape = env_info["state_shape"]
obs_shape = env_info["obs_shape"]
episode_limit = env_info["episode_limit"]

## Batch Replay Settings
minibatch_size = 128
epoch = 1
minimum_batch_size = 1024 * 4

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Progress bar
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Network Setup
atoms = 256
network = Network(
    state_shape=state_shape,
    obs_shape=obs_shape,
    action_space=action_space,
    atoms=atoms,
    save_path=MODEL_PATH,
)

# Resotre / Initialize
global_episodes = network.initiate()
print(global_episodes)

writer = tf.summary.create_file_writer(LOG_PATH)

# TRAINING 
def train_central(
    network,
    trajs,
    bootstrap=0.0,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(trajs):
        # Forward
        states = np.array(traj[0])
        last_state = np.array(traj[3])[-1:, ...]
        reward = traj[2]

        env_critic, _ = network.run_network_central(states)
        _env_critic, _ = network.run_network_central(last_state)
        critic = env_critic["critic"].numpy()[:, 0].tolist()
        _critic = _env_critic["critic"].numpy()[0, 0]

        # TD-target
        td_target_c, _ = gae(reward, critic, _critic, gamma, lambd, discount_adv=False)

        traj_buffer["state"].extend(traj[0])
        traj_buffer["td_target_c"].extend(td_target_c)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]).astype(np.float32),
                "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .repeat(epoch)
        .batch(batch_size)
    )

    network.update_central(
        train_dataset, writer=writer, log=log, step=step, tag="losses/"
    )

def train_decentral(
    agent_trajs,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    # Agent trajectory processing
    traj_buffer = defaultdict(list)
    advantage_lists = []
    f1_list = []
    f2_list = []
    fc_list = []
    for trajs in agent_trajs:
        for idx, traj in enumerate(trajs):
            reward = traj[2]
            mask = traj[3]
            critic = traj[5]
            phi = np.array(traj[7]).tolist()
            psi = np.array(traj[8]).tolist()
            _critic = traj[9][-1]
            _psi = np.array(traj[10][-1])

            cent_state = np.array(traj[14])
            env_critic, _ = network.run_network_central(cent_state)
            env_critic = env_critic["critic"].numpy()[:, 0].tolist()
            #cent_last_state = np.array(traj[15])[-1:, ...]
            #_env_critic, _ = network.run_network_central(cent_last_state)
            #_env_critic = _env_critic["critic"].numpy()[0, 0]

            icritic = traj[12]

            dc = np.array(critic[1:])-np.array(critic[:-1])
            dc1 = np.array(env_critic[1:])-np.array(env_critic[:-1])
            dc2 = np.array(icritic[1:])-np.array(icritic[:-1])
            f1_list.append(np.mean((dc * dc1)>0))
            f2_list.append(np.mean((dc * dc2)>0))
            fc_list.append(np.mean((dc * dc)>0))

            # Zero bootstrap because all trajectory terminates
            td_target_c, advantages_global = gae(
                reward, critic, _critic,
                gamma, lambd, # mask=mask,
                normalize=False
            )
            _, advantages = gae(
                traj[11],
                traj[12],
                traj[13][-1],
                gamma,
                lambd,
               # mask=mask,
                normalize=False,
            )
            '''
            _, advantages = gae(
                reward,
                env_critic,
                _env_critic,
                gamma,
                lambd,
                normalize=False,
            )
            '''
            td_target_psi, _ = gae(
                phi,
                psi,
                _psi,  # np.zeros_like(phi[0]),
                gamma,
                lambd,
               # mask=np.array(mask)[:, None],
                discount_adv=False,
                normalize=False,
            )
            beta = max(min((-0.9/30000)*step + 1, 1.0),0.1)

            traj_buffer["state"].extend(traj[0])
            traj_buffer["next_state"].extend(traj[4])
            traj_buffer["log_logit"].extend(traj[6])
            traj_buffer["action"].extend(traj[1])
            traj_buffer["old_value"].extend(critic)
            traj_buffer["td_target_psi"].extend(td_target_psi)
            traj_buffer["advantage"].extend(advantages)
            traj_buffer["td_target_c"].extend(td_target_c)
            traj_buffer["rewards"].extend(reward)

    train_datasets = []
    num_type = 1
    for atype in range(num_type):
        #traj_buffer = traj_buffer_list[atype]
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer["state"]).astype(np.float32),
                    "old_log_logit": np.stack(traj_buffer["log_logit"]).astype(np.float32),
                    "action": np.stack(traj_buffer["action"]),
                    "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
                    "td_target": np.stack(traj_buffer["td_target_psi"]).astype(np.float32),
                    "advantage": np.stack(traj_buffer["advantage"]).astype(np.float32),
                    "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                    "rewards": np.stack(traj_buffer["rewards"]).astype(np.float32),
                    "next_state": np.stack(traj_buffer["next_state"]).astype(np.float32),
                }
            )
            .shuffle(64)
            .repeat(epoch)
            .batch(batch_size)
        )
        train_datasets.append(train_dataset)
        
    network.update_decentral(
        train_datasets, writer=writer, log=log, step=step, tag="losses/", 
    )
    if log:
        with writer.as_default():
            tag = "advantages/"
            tf.summary.scalar('beta/adv_beta', beta, step=step)
            for idx, adv_list in enumerate(advantage_lists):
                tb_log_histogram(
                    np.array(adv_list), tag + "dec_advantages_{}".format(idx), step=step
                )
            tf.summary.scalar('factoredness/f1', np.mean(f1_list), step=step)
            tf.summary.scalar('factoredness/f2', np.mean(f2_list), step=step)
            tf.summary.scalar('factoredness/fc', np.mean(fc_list), step=step)
            writer.flush()


def run_network(observations, env):
    # Get available actions
    avail_actions = []
    for i in range(num_agent):
        avail_action = env.get_avail_agent_actions(i)
        avail_actions.append(avail_action)
    avail_actions = np.asarray(avail_actions)

    # Run decentral network
    observations = np.asarray(observations)
    actor, critic = network.run_network_decentral(observations)
    # dec_results[0] --> actor. actor['log_softmax'] --> tf.random.categorical

    # Get action
    action_probs = actor['softmax'].numpy() * avail_action
    a1 = [np.random.choice(action_space, p=p/p.sum()) for p in action_probs]

    # Container
    '''
    a1 = np.empty([num_agent], dtype=np.int32)
    vg1 = np.empty([num_agent], dtype=np.float32)
    vc1 = np.empty([num_agent], dtype=np.float32)
    phi1 = np.empty([num_agent, atoms], dtype=np.float32)
    psi1 = np.empty([num_agent, atoms], dtype=np.float32)
    log_logits1 = np.empty([num_agent, action_space], dtype=np.float32)
    reward_pred1 = np.empty([num_agent], dtype=np.float32)
    '''

    # Postprocessing
    '''
    for (a, actor, critic), mask in zip(results, agent_type_masking):
        vg = critic["critic"]
        vc = critic["icritic"]
        phi = critic["phi"]
        psi = critic["psi"]
        log_logits = actor["log_softmax"]
        reward_pred = critic["reward_predict"]

        a1[mask] = a
        vg1[mask] = vg
        vc1[mask] = vc
        phi1[mask, :] = phi
        psi1[mask, :] = psi
        log_logits1[mask, :] = log_logits
        reward_pred1[mask] = reward_pred
    action = np.reshape(a1, [NENV, num_blue])
    '''
    vg1 = critic["critic"]
    vc1 = critic["icritic"]
    phi1 = critic["phi"]
    psi1 = critic["psi"]
    log_logits1 = actor["log_softmax"]
    reward_pred1 = critic["reward_predict"]

    return a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1


batch = []
batch_size = 0
dec_batch = []
dec_batch_size = 0
while global_episodes < total_episodes:
#while True:
    # initialize parameters
    dec_trajs = [Trajectory(depth=16) for _ in range(num_agent)]
    cent_trajs = Trajectory(depth=4)

    # Reset Game
    env.reset()
    done = False
    episode_reward = 0

    # Bootstrap
    s1 = env.get_state()
    o1 = env.get_obs()
    a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(o1, env)

    # Rollout
    stime_roll = time.time()
    step = 0
    while not done:
        s0 = s1
        o0 = o1
        a0 = a1
        vg0 = vg1
        vc0 = vc1
        psi0 = psi1
        phi0 = phi1
        log_logits0 = log_logits1
        was_done = done

        # Run environment
        reward, done, _ = env.step(a0)
        step += 1
        episode_reward += reward

        # Run network
        s1 = env.get_state()
        o1 = env.get_obs()
        a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(o1, env)

        # Experience recording
        for idx in range(num_agent):
            dec_trajs[idx].append(
                [
                    o0[idx],
                    a0[idx],
                    reward,  # + reward_pred1[idx], # Advantage
                    done[env_idx],  # masking
                    o1[idx],
                    vg0[idx],  # Advantage
                    log_logits0[idx],  # PPO
                    phi0[idx],  # phi: one-step ahead
                    psi0[idx],
                    vg1[idx],
                    psi1[idx],
                    reward,
                    #reward[env_idx]-(reward_pred1[idx] if reward[env_idx] else 0),
                    #reward[env_idx]-reward_pred1[idx],
                    vc0[idx],
                    vc1[idx],
                    s0,
                    s1,
                ]
            )
        cent_trajs.append([
            s0,
            reward,
            done,
            s1,
        ])
    etime_roll = time.time()

    if PRINT:
        print("Reward in episode {} = {}".format(global_episodes, episode_reward))

    # decentralize training
    dec_batch.extend(dec_trajs)
    dec_batch_size += len(dec_trajs) * step
    if dec_batch_size > minimum_batch_size:
        stime_train = time.time()
        log = interval_flag(global_episodes, save_image_frequency, "im_log")
        train_decentral(
            dec_batch,
            epoch=epoch,
            batch_size=minibatch_size,
            writer=writer,
            log=log,
            step=global_episodes,
        )
        dec_batch = []
        dec_batch_size = 0
        etime_train = time.time()
        log_traintime.append(etime_train - stime_train)

    # centralize training
    batch.extend(cent_trajs)
    batch_size += step
    if batch_size >= minimum_batch_size // 2:
        log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
        train_central(network, batch, 0, epoch, minibatch_size, writer, log_tc_on, global_episodes)
        batch = []
        batch_size = 0

    log_episodic_reward.extend(episode_rew.tolist())
    log_looptime.append(etime_roll - stime_roll)

    # Stepper
    global_episodes += 1
    if PROGBAR:
        progbar.update(global_episodes)

    # Log
    log_on = interval_flag(global_episodes, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "env_reward", log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag + "rollout_time", log_looptime(), step=global_episodes)
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_episodes)
            writer.flush()

    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)

    # Save Gameplay
    log_save_analysis = interval_flag(global_episodes, 1024 * 4, "save_log")
    if log_save_analysis:
        pass
