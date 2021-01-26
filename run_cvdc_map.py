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

import time
import gym
import numpy as np
import random
import math
from collections import defaultdict
from functools import partial

import multiagent
from MAPredatorPreyWrappers import RandomPreyActions,MAPFrameStacking,PredatorPreyTerminator

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.logger import *
from utility.gae import gae

# from utility.slack import SlackAssist

from method.CVDC import SF_CVDC as Network

parser = argparse.ArgumentParser(description="CVDC(learnability) PredPrey")
parser.add_argument("--train_number", type=int, help="training train_number")
parser.add_argument("--machine", type=str, help="training machine")
parser.add_argument("--silence", action="store_false", help="call to disable the progress bar")
parser.add_argument("--print", action="store_true", help="print out the progress in detail")
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--training_episodes", type=int, default=10000000, help='number of training episodes')
parser.add_argument("--gpu", action="store_false", help='Use of the GPU')
args = parser.parse_args()

if args.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PROGBAR = args.silence
PRINT = args.print

## Training Directory Reset
TRAIN_NAME = "CVDC_PredPrey_{}_{:02d}".format(
    args.machine,
    args.train_number,
)
#slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")
TRAIN_TAG = "CVDC(learnability), " + TRAIN_NAME
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
gamma = 0.99  # GAE - discount
lambd = 0.95  # GAE - lambda

# Log
save_network_frequency = 4096
save_stat_frequency = 2000
save_image_frequency = 2000
moving_average_step = 2000 # MA for recording episode statistics

## Environment
frame_stack = 2
env = gym.make("PredPrey-v0")

env_info = env.get_env_info()
# env = SMACWrapper(env)
env = RandomPreyActions(env)
env = MAPFrameStacking(env, numFrames=frame_stack)
env = PredatorPreyTerminator(env)
print(env_info)
# exit()

# Environment/Policy Settings
action_space = env_info["n_actions"].n
num_agent = env_info["n_agents"]-1
state_shape = [frame_stack, env.state_space.shape[0]]#env_info["state_shape"] * frame_stack
obs_shape = [frame_stack, env.observation_space.shape[0]]#env_info["obs_shape"] * frame_stack
episode_limit = env_info["episode_limit"]

## Batch Replay Settings
minibatch_size = 256
epoch = 1
buffer_size = 2048
drop_remainder = False

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Progress bar
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Network Setup
atoms = 128
network = Network(
    state_shape=state_shape,
    obs_shape=obs_shape,
    action_space=action_space,
    atoms=atoms,
    save_path=MODEL_PATH,
    lr=5E-5,
    clr=5E-5,
    entropy=0.00,
)
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
        reward = traj[1]

        env_critic, _ = network.run_network_central(states)
        _env_critic, _ = network.run_network_central(last_state)
        critic = env_critic["critic"].numpy().tolist()
        _critic = _env_critic["critic"].numpy()

        # TD-target
        td_target_c, _ = gae(reward, critic, _critic, gamma, lambd,
            normalize=False,
            td_lambda=True
        )

        traj_buffer["state"].extend(traj[0])
        traj_buffer["td_target_c"].extend(td_target_c)
        traj_buffer["old_value"].extend(critic)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]).astype(np.float32),
                "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .batch(batch_size, drop_remainder=True)
    )

    network.update_central(
        train_dataset, epoch, writer=writer, log=log, step=step, tag="losses/"
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
    for traj in agent_trajs:
        reward = traj[2]
        mask = traj[3]
        critic = traj[5]
        phi = np.array(traj[7]).tolist()
        psi = np.array(traj[8]).tolist()
        _critic = traj[9][-1]
        _psi = np.array(traj[10][-1])

        cent_state = np.array(traj[14])
        env_critic, _ = network.run_network_central(cent_state)
        env_critic = env_critic["critic"].numpy().tolist()
        cent_last_state = np.array(traj[15])[-1:, ...]
        _env_critic, _ = network.run_network_central(cent_last_state)
        _env_critic = _env_critic["critic"].numpy()

        icritic = traj[12]
        mask = traj[3]

        '''
        dc = np.array(critic[1:])-np.array(critic[:-1])
        dc1 = np.array(env_critic[1:])-np.array(env_critic[:-1])
        dc2 = np.array(icritic[1:])-np.array(icritic[:-1])
        f1_list.append(np.mean((dc * dc1)>0))
        f2_list.append(np.mean((dc * dc2)>0))
        fc_list.append(np.mean((dc * dc)>0))
        '''

        # Zero bootstrap because all trajectory terminates
        td_target_c, advantages_global = gae(
            reward, critic, _critic,
            gamma, lambd, normalize=False,
            td_lambda=True, # mask=mask,
            standardize_td=True,
        )
        _, advantages_global = gae(
            reward,
            env_critic,
            _env_critic,
            gamma,
            lambd,
            normalize=False,
            td_lambda=True,
            #mask=mask,
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
        td_target_psi, _ = gae(
            phi,
            psi,
            _psi,  # np.zeros_like(phi[0]),
            gamma,
            lambd,
           # mask=np.array(mask)[:, None],
            discount_adv=False,
            normalize=False,
            td_lambda=False,
        )

        '''
        traj_length = np.where(mask)[0]
        if len(traj_length) == 0:
            traj_length = len(traj[0])
        else:
            traj_length = traj_length[0]
        '''
        traj_buffer["state"].extend(traj[0])
        traj_buffer["next_state"].extend(traj[4])
        traj_buffer["log_logit"].extend(traj[6])
        traj_buffer["action"].extend(traj[1])
        traj_buffer["old_value"].extend(critic)
        traj_buffer["td_target_psi"].extend(td_target_psi)
        traj_buffer["advantage"].extend(advantages_global)
        traj_buffer["td_target_c"].extend(td_target_c)
        traj_buffer["rewards"].extend(reward)
        traj_buffer["avail_actions"].extend(traj[16])

    train_datasets = []  # Each for type of agents
    num_type = 1
    for atype in range(num_type):
        #traj_buffer = traj_buffer_list[atype]

        # Normalize Advantage (global)
        _adv = np.array(traj_buffer["advantage"]).astype(np.float32)
        _adv = (_adv - _adv.mean()) / (_adv.std()+1e-9)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer["state"]).astype(np.float32),
                    "old_log_logit": np.stack(traj_buffer["log_logit"]).astype(np.float32),
                    "action": np.stack(traj_buffer["action"]),
                    "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
                    "td_target": np.stack(traj_buffer["td_target_psi"]).astype(np.float32),
                    "advantage": _adv,
                    "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                    "rewards": np.stack(traj_buffer["rewards"]).astype(np.float32),
                    "next_state": np.stack(traj_buffer["next_state"]).astype(np.float32),
                    "avail_actions": np.stack(traj_buffer["avail_actions"]).astype(np.bool)
                }
            )
            .shuffle(64)
            .batch(batch_size, drop_remainder=drop_remainder)
        )
        train_datasets.append(train_dataset)
    network.update_decentral(
        train_datasets, epoch, writer=writer, log=log, step=step, tag="losses/",
    )
    if log:
        with writer.as_default():
            tag = "advantages/"
            for idx, adv_list in enumerate(advantage_lists):
                tb_log_histogram(
                    np.array(adv_list), tag + "dec_advantages_{}".format(idx), step=step
                )
            #tf.summary.scalar('factoredness/f1', np.mean(f1_list), step=step)
            #tf.summary.scalar('factoredness/f2', np.mean(f2_list), step=step)
            #tf.summary.scalar('factoredness/fc', np.mean(fc_list), step=step)
            writer.flush()


def run_network(observations, env):
    # Get available actions
    avail_actions = []
    for i in range(num_agent):
        avail_action = env.get_avail_agent_actions(i)
        avail_actions.append(avail_action)
    avail_actions = np.array(avail_actions)

    # Run decentral network
    observations = np.asarray(observations)
    actor, critic = network.run_network_decentral(observations, avail_actions)
    # dec_results[0] --> actor. actor['log_softmax'] --> tf.random.categorical

    # Get action
    probs = actor['softmax'].numpy()
    action_probs = probs * avail_actions
    try:
        a1 = []
        for idx, p in enumerate(action_probs):
            if np.isclose(p.sum(), 0):
                avail_actions = env.get_avail_agent_actions(idx)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                a1.append(action)
            else:
                p = p/p.sum()
                action = np.random.choice(action_space, p=p)
                a1.append(action)
    except ValueError:
        print(probs)
        print(action_probs)
        print(observations)
        raise ValueError

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

    return a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, avail_actions


while global_episodes < total_episodes:
#while True:
    batch = []
    batch_size = 0
    dec_batch = []
    dec_batch_size = 0
    while dec_batch_size < buffer_size:
        # initialize parameters
        dec_trajs = [Trajectory(depth=17) for _ in range(num_agent)]
        cent_trajs = Trajectory(depth=4)

        # Reset Game
        env.reset()
        done = np.array([False]*num_agent)
        episode_reward = 0

        # Bootstrap
        s1 = env.get_state()
        o1 = env.get_obs()
        a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, _avail_actions = run_network(o1, env)

        # Rollout
        stime_roll = time.time()
        step = 0
        count = 0
        while not done.all():
            s0 = s1
            o0 = o1
            a0 = a1
            vg0 = vg1
            vc0 = vc1
            psi0 = psi1
            phi0 = phi1
            log_logits0 = log_logits1
            was_done = done
            avail_actions = _avail_actions

            # Run environment
            #reward, done, _ = env.step(a0)
            o1, reward, done, info, s1 = env.step(a0)
            step += 1
            episode_reward += reward

            # Run network
            a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, _avail_actions = run_network(o1, env)
            if done.all()and count==0:
                continue
            else:
                count+=1

                # Experience recording
                for idx in range(num_agent):
                    if np.all(o0[idx]==0):
                        continue
                    dec_trajs[idx].append(
                        [
                            o0[idx],
                            a0[idx],
                            reward,  # + reward_pred1[idx], # Advantage
                            was_done[idx],  # masking
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
                            s0[0],
                            s1[0],
                            avail_actions[idx]
                        ]
                    )
                cent_trajs.append([
                    s0[0],
                    reward,
                    was_done.all(),
                    s1[0],
                ])
        etime_roll = time.time()

        log_episodic_reward.append(episode_reward)
        log_looptime.append(etime_roll - stime_roll)

        # Buffer
        if count < 1:
            # print("Here")
            pass
        else:
            log_winrate.append(count)
            dec_batch.extend(dec_trajs)
            dec_batch_size += sum([len(traj) for traj in dec_trajs])
            batch.append(cent_trajs)
            batch_size += step

            if PRINT:
                print("Reward in episode {} = {} {}".format(global_episodes, episode_reward, TRAIN_NAME))

            # Stepper
            global_episodes += step
            if PROGBAR:
                progbar.update(global_episodes)

    # decentralize training
    if PRINT:
        print('training: {}'.format(dec_batch_size))
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
    etime_train = time.time()
    log_traintime.append(etime_train - stime_train)

    # centralize training
    if PRINT:
        print('training(central): {}'.format(batch_size))
    log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
    train_central(
        network,
        batch,
        0,
        epoch,
        minibatch_size,
        writer,
        log_tc_on,
        global_episodes
    )

    # Log
    log_on = interval_flag(global_episodes, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "env_reward", log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag + "episode_length", log_winrate(), step=global_episodes)
            tf.summary.scalar(tag + "rollout_time", log_looptime(), step=global_episodes)
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_episodes)
            writer.flush()

    # Network Save
    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)

    # Save Gameplay
    log_save_analysis = interval_flag(global_episodes, 1024 * 4, "save_log")
    if log_save_analysis:
        pass
