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

import yaml

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
from SC2Wrappers import SMACWrapper

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
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
parser.add_argument("--seed", type=int, default=-1, help='random seed (-1 for no seed)')
parser.add_argument("--step_mul", type=int, default=8, help='how many steps to make an action')
parser.add_argument("--training_steps", type=int, default=100000000, help='number of training episodes')
parser.add_argument("--config", type=str, default=None, help='configuration file location')
parser.add_argument("--entropy_beta", type=float, default=None, help='entropy beta')
args = parser.parse_args()

## Training Directory Reset
ALG_NAME = "CVDC"
TRAIN_NAME = "{}_{}_{}_{:02d}".format(
    ALG_NAME,
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

# Import configuration
if args.config is not None:
    # Model path given
    config_path = args.config
elif os.path.exists(os.path.join(MODEL_PATH, 'config.yaml')):
    # Saved configuration
    config_path = os.path.join(MODEL_PATH, 'config.yaml')
elif os.path.exists(os.path.join('config','CVDC_'+args.map+'.yaml')):
    # Model path in config directory
    config_path = os.path.join('config', 'CVDC_'+args.map+'.yaml')
else:
    # Default config
    config_path = os.path.join('config', 'CVDC_default.yaml')
param = yaml.safe_load(open('config/CVDC_default.yaml', 'r'))
param.update(yaml.safe_load(open(config_path, 'r')))

# Customize parameter
if args.entropy_beta is not None:
    param['entropy beta'] = args.entropy_beta

# Random Seeding
seed = args.seed
if seed != -1:
    tf.random.set_seed(seed)
    np.random.seed(seed)

PROGBAR = args.silence
PRINT = args.print

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

with open(os.path.join(MODEL_PATH, 'config.yaml'), 'w') as file:
    yaml.dump(param, file)

# Training
total_steps = args.training_steps
gamma = param['gamma']  # GAE - discount
lambd = param['lambda']  # GAE - lambda

# Log
save_network_frequency = param['save model frequency']
save_stat_frequency = param['save statistic frequency']
save_image_frequency = param['save image frequency']
moving_average_step = param['moving average step']
test_interval = param['test interval']
test_nepisode = param['test episode']

## Environment
frame_stack = param['frame stack']
env = StarCraft2Env(
    map_name=args.map,
    step_mul=args.step_mul,
    difficulty=args.difficulty,
    replay_dir=SAVE_PATH
)
env = SMACWrapper(env, numFramesObs=frame_stack, lstm=True)
env_info = env.env_info
print(env_info)
        
# Environment/Policy Settings
action_space = env_info["n_actions"]
num_agent = env_info["n_agents"]
obs_shape = [frame_stack, env_info['obs_shape'] + action_space + num_agent] 
state_shape = [env_info['state_shape']]
episode_limit = env_info["episode_limit"]

## Batch Replay Settings
minibatch_size = param['minibatch size']
epoch = param['epoch'] 
buffer_size = param['buffer size']
drop_remainder = param['drop remainder']

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Progress bar
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Network Setup
atoms = param['atoms']
network = Network(
    state_shape=state_shape,
    obs_shape=obs_shape,
    action_space=action_space,
    num_agent=num_agent,
    num_action=action_space,
    save_path=MODEL_PATH,
    atoms=atoms,
    param=param,
)
global_steps = network.initiate()
print(global_steps)
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
    update_target=False,
):
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(trajs):
        # Forward
        states = np.array(traj[0])
        last_state = np.array(traj[3])[-1:, ...]
        inputs = np.array(traj[4])
        last_inputs = np.array(traj[5])[-1:, ...]
        reward = traj[1]

        env_critic, _ = network.run_network_central_target(inputs, states)
        _env_critic, _ = network.run_network_central_target(last_inputs, last_state)
        critic = list(env_critic["critic"].numpy())
        _critic = _env_critic["critic"].numpy()[0]

        # TD-target
        td_target_c, _ = gae(
            reward,
            critic,
            _critic,
            gamma,
            lambd,
            td_lambda=param['central critic td lambda'],
            standardize_td=param['central critic td standardize'],
        )

        traj_buffer["state"].extend(traj[0])
        traj_buffer["td_target_c"].extend(td_target_c)
        traj_buffer["old_value"].extend(critic)
        traj_buffer["inputs"].extend(traj[4])

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]).astype(np.float32),
                "inputs": np.stack(traj_buffer["inputs"]).astype(np.float32),
                "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .batch(batch_size, drop_remainder=True)
        .repeat(epoch)
    )

    network.update_central(
        train_dataset, writer=writer, log=log, step=step, tag="losses/", param=param,
    )

    if update_target:
        network.update_target()

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
        _critic = traj[9][-1]
        phi = list(np.array(traj[7]))
        psi = list(np.array(traj[8]))
        _psi = np.array(traj[10][-1])

        cent_state = np.array(traj[14])
        cent_inputs = np.array(traj[17])
        env_critic, _ = network.run_network_central(cent_inputs, cent_state)
        env_critic = list(env_critic["critic"].numpy())
        cent_last_state = np.array(traj[15])[-1:, ...]
        cent_last_inputs = np.array(traj[18])[-1:,...]
        _env_critic, _ = network.run_network_central(cent_last_inputs, cent_last_state)
        _env_critic = _env_critic["critic"].numpy()[0]

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
            gamma, lambd,
            td_lambda=param['decentral critic td lambda'],
            standardize_td=param['decentral critic td standardize'],
            #mask=mask,
        )
        _, advantages_global = gae(
            reward,
            env_critic,
            _env_critic,
            gamma,
            lambd,
            #mask=mask,
        )
        _, advantages = gae(
            traj[11],
            traj[12],
            traj[13][-1],
            gamma,
            lambd,
            #mask=mask,
        )
        td_target_psi, _ = gae(
            phi,
            psi,
            _psi,  # np.zeros_like(phi[0]),
            gamma,
            lambd,
            discount_adv=False, # just to save time
            td_lambda=False,
            #mask=np.array(mask)[:, None],
        )

        '''
        traj_length = np.where(mask)[0]
        if len(traj_length) == 0:
            traj_length = len(traj[0])
        else:
            traj_length = traj_length[0]
        traj_length = np.where(mask)[0]
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
        if param['advantage global norm']:
            _adv = (_adv - _adv.mean()) / (_adv.std()+1e-9)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer["state"]).astype(np.float32),
                    "old_log_logit": np.stack(traj_buffer["log_logit"]).astype(np.float32),
                    "action": np.stack(traj_buffer["action"]),
                    "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
                    "td_target_psi": np.stack(traj_buffer["td_target_psi"]).astype(np.float32),
                    "advantage": _adv,
                    "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                    "rewards": np.stack(traj_buffer["rewards"]).astype(np.float32),
                    "next_state": np.stack(traj_buffer["next_state"]).astype(np.float32),
                    "avail_actions": np.stack(traj_buffer["avail_actions"]).astype(np.bool)
                }
            )
            .shuffle(64)
            .batch(batch_size, drop_remainder=drop_remainder)
            .repeat(epoch)
        )
        train_datasets.append(train_dataset)
        
    network.update_decentral(
        train_datasets, writer=writer, log=log, step=step, tag="losses/", param=param,
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


def run_network(observations, avail_actions, argmax_policy=False):
    # Run decentral network
    observations = np.asarray(observations)
    actor, critic = network.run_network_decentral(observations, avail_actions)
    # dec_results[0] --> actor. actor['log_softmax'] --> tf.random.categorical

    # Get action
    probs = actor['softmax'].numpy()
    action_probs = probs * avail_actions # (TODO) this process should not be necessary
    try:
        if not argmax_policy:
            a1 = []
            for idx, p in enumerate(action_probs):
                if np.isclose(np.abs(p).sum(), 0): # too close to zero. random action
                    avail_actions = avail_actions[idx]
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
                    a1.append(action)
                else: # stochastic action
                    p = p/p.sum()
                    action = np.random.choice(action_space, p=p)
                    a1.append(action)
        elif argmax_policy:
            a1 = np.argmax(action_probs, axis=1)
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
    vg1 = critic["critic"].numpy()
    vc1 = critic["icritic"].numpy()
    phi1 = critic["phi"].numpy()
    psi1 = critic["psi"].numpy()
    log_logits1 = actor["log_softmax"].numpy()
    reward_pred1 = critic["reward_predict"].numpy()

    return a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 

def rollout(n_episode=None, argmax_policy=False, max_step=1e9):
    # if n_episode is none, run upto max_step
    stime = time.time()
    episode = 0 
    step = 0
    reward_record = []
    win_record = []
    while step < max_step:
        episode += 1
        if n_episode is not None and episode >= n_episode:
            break

        # Reset Game
        o1, s1 = env.reset()
        _avail_actions = env.get_avail_actions()
        done = [False] * num_agent
        episode_reward = 0

        # Rollout
        while not all(done):
            action = run_network(o1, _avail_actions, argmax_policy=argmax_policy)[0]
            (o1, s1), reward, done, info = env.step(action)
            _avail_actions = info['valid_action']
            episode_reward += reward
            step += 1

        # Log
        reward_record.append(episode_reward)
        win_record.append(info['battle_won'])
    rollout_walltime = time.time() - stime

    return np.array(reward_record), np.array(win_record)

global_episodes = 0
while global_steps < total_steps:
    batch = []
    batch_size = 0
    dec_batch = []
    dec_batch_size = 0
    episode = 0
    _rollout_rewards = []
    #while dec_batch_size < buffer_size and episode < 8:
    while episode < 8:
        global_episodes += 1
        episode += 1

        # initialize parameters
        dec_trajs = [Trajectory(depth=19) for _ in range(num_agent)]
        cent_trajs = Trajectory(depth=6)

        # Reset Game
        o1, s1 = env.reset()
        _avail_actions = env.get_avail_actions()
        done = [False] * num_agent
        episode_reward = 0

        # Bootstrap
        a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(o1, _avail_actions)

        # Rollout
        stime_roll = time.time()
        step = 0
        while not all(done):
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
            (o1, s1), reward, done, info = env.step(a0)
            _avail_actions = info['valid_action']
            episode_reward += reward
            step += 1

            # Run network
            a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(o1, _avail_actions)

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
                        #reward,
                        reward-(reward_pred1[idx] if reward else 0),
                        vc0[idx],
                        vc1[idx],
                        s0,
                        s1,
                        avail_actions[idx],
                        vg0,
                        vg1,
                    ]
                )
            cent_trajs.append([
                s0,
                reward,
                np.any(was_done),
                s1,
                vg0,
                vg1,
            ])
        etime_roll = time.time()
        
        log_episodic_reward.append(episode_reward)
        log_looptime.append(etime_roll - stime_roll)

        battle_won = info['battle_won']
        log_winrate.append(battle_won)

        # Buffer
        dec_batch.extend(dec_trajs)
        dec_batch_size += sum([len(traj) for traj in dec_trajs])
        batch.append(cent_trajs)
        batch_size += step

        _rollout_rewards.append(episode_reward)

        # Stepper
        global_steps += step
        if PROGBAR:
            progbar.update(global_steps)

    if PRINT:
        print("Reward in episode batch {} = {} {}".format(global_steps, np.mean(_rollout_rewards), TRAIN_NAME))

    # decentralize training
    if PRINT:
        print('training: {}'.format(dec_batch_size))
    stime_train = time.time()
    log = interval_flag(global_steps, save_image_frequency, "im_log")
    train_decentral(
        dec_batch,
        epoch=epoch,
        batch_size=minibatch_size,
        writer=writer,
        log=log,
        step=global_steps,
    )
    etime_train = time.time()
    log_traintime.append(etime_train - stime_train)

    # centralize training
    if PRINT:
        print('training(central): {}'.format(batch_size))
    log_tc_on = interval_flag(global_steps, save_image_frequency, 'tc_log')
    update_target_flag = interval_flag(global_episodes, param['update target interval'], 'target_interv')
    train_central(
        network, 
        batch, 
        0, 
        epoch, 
        minibatch_size, 
        writer, 
        log_tc_on,
        global_steps,
        update_target_flag,
    )

    # Test Run
    test_run_on = interval_flag(global_steps, test_interval, "test")
    if test_run_on:
        test_rewards, test_winrates = rollout(test_nepisode, argmax_policy=True)
        if PRINT:
            print('Reward in test batch = {}, winrate: {}'.format(test_rewards.mean(), test_winrates.mean()))
        with writer.as_default():
            tag = "test"
            tf.summary.scalar(tag + "reward_mean", test_rewards.mean(), step=global_steps)
            tf.summary.scalar(tag + "reward_std", test_rewards.std(), step=global_steps)
            tf.summary.scalar(tag + "winrate_mean", test_winrates.mean(), step=global_steps)
            writer.flush()

    # Log
    log_on = interval_flag(global_steps, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "env_reward", log_episodic_reward(), step=global_steps)
            tf.summary.scalar(tag + "winrate", log_winrate(), step=global_steps)
            tf.summary.scalar(tag + "rollout_time", log_looptime(), step=global_steps)
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_steps)
            writer.flush()

    # Network Save
    save_on = interval_flag(global_steps, save_network_frequency, "save")
    if save_on:
        network.save(global_steps)

    # Save Gameplay
    log_save_analysis = interval_flag(global_steps, 1024 * 4, "save_log")
    if log_save_analysis:
        pass

