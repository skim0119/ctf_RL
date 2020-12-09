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

from method.CVDC2 import SF_CVDC_SC2 as Network

from smac.env import StarCraft2Env

parser = argparse.ArgumentParser(description="PPO trainer for convoy")
parser.add_argument("--train_number", type=int, help="training train_number",default = 1)
parser.add_argument("--machine", type=str, help="training machine",default = "Neale")
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
keep_frame = 1
nchannel = 6 * keep_frame
# Batch Replay Settings
minibatch_size = 128
epoch = 2
minimum_batch_size = 4096
num_agent = 8

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
cent_input_size = [None,envs.env.get_state_size()]

# envs = [make_env() for i in range(NENV)]
# envs = SubprocVecEnv(envs, keep_frame=keep_frame, size=vision_dx)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Agent Type Setup
agent_type = [8]
num_type = len(agent_type)
agent_type_masking = np.zeros([num_type, 8], dtype=bool)
agent_type_index = np.zeros([8], dtype=int)
prev_i = 0
for idx, i in enumerate(np.cumsum(agent_type)):
    agent_type_masking[idx, prev_i:i] = True
    agent_type_index[prev_i:i] = idx
    prev_i = i
agent_type_masking = np.tile(agent_type_masking, NENV)

# Network Setup
atoms = 256
network = Network(
    central_obs_shape=cent_input_size,
    decentral_obs_shape=input_size,
    action_size=action_space,
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
        last_state = np.array(traj[3])[-1:, :, :, :]
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
    log_image=False,
):
    train_datasets = []

    # Agent trajectory processing
    traj_buffer_list = [defaultdict(list) for _ in range(num_type)]
    advantage_lists = [[] for _ in range(num_type)]
    f1_list = []
    f2_list = []
    fc_list = []
    for trajs in agent_trajs:
        for idx, traj in enumerate(trajs):
            atype = agent_type_index[idx]

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

            traj_buffer = traj_buffer_list[atype]
            traj_buffer["state"].extend(traj[0])
            traj_buffer["next_state"].extend(traj[4])
            traj_buffer["log_logit"].extend(traj[6])
            traj_buffer["action"].extend(traj[1])
            traj_buffer["old_value"].extend(critic)
            traj_buffer["td_target_psi"].extend(td_target_psi)
            traj_buffer["advantage"].extend(advantages)
            traj_buffer["td_target_c"].extend(td_target_c)
            traj_buffer["rewards"].extend(reward)
    for atype in range(num_type):
        traj_buffer = traj_buffer_list[atype]
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
        train_datasets, writer=writer, log=log, step=step, tag="losses/", log_image=log_image,
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


def run_network(states,validActions):
    # State Process
    states_list = []
    for mask in agent_type_masking:
        state = np.compress(mask, states, axis=0)
        states_list.append(state)

    # Run network
    results = network.run_network_decentral(states_list,validActions)

    # Container
    a1 = np.empty([NENV * num_agent], dtype=np.int32)
    vg1 = np.empty([NENV * num_agent], dtype=np.float32)
    vc1 = np.empty([NENV * num_agent], dtype=np.float32)
    phi1 = np.empty([NENV * num_agent, atoms], dtype=np.float32)
    psi1 = np.empty([NENV * num_agent, atoms], dtype=np.float32)
    log_logits1 = np.empty([NENV * num_agent, action_space], dtype=np.float32)
    reward_pred1 = np.empty([NENV * num_agent], dtype=np.float32)

    # Postprocessing
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
    action = np.reshape(a1, [num_agent])
    return action, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1


batch = []
dec_batch = []
while global_episodes < total_episodes:
#while True:
    # Flags
    log_save_analysis = False  #interval_flag(global_episodes, 1024 * 4, "save_log")

    # initialize parameters
    episode_rew = np.zeros(NENV)
    is_alive = [not va[0] for va in envs.get_avail_actions()]
    is_done = [False for env in range(NENV * num_agent)]

    trajs = [[Trajectory(depth=16) for _ in range(num_agent)] for _ in range(NENV)]
    cent_trajs = [Trajectory(depth=4) for _ in range(NENV)]

    # Bootstrap

    s1 = envs.reset()
    s1 = np.stack(s1).astype(np.float32)
    validActions = envs.get_avail_actions()

    # actions, a1, v1, p1 = get_action(s1,validActions)

    cent_s1 = np.expand_dims(envs.env.get_state().astype(np.float32),0)  # Centralized
    # cent_s1 = envs.env.get_state().astype(np.float32)  # Centralized

    actions, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(s1,validActions)

    reward_pred_list = []

    # Rollout
    stime_roll = time.time()
    _states = [[] for _ in range(NENV)]
    _agent1_r = [[] for _ in range(NENV)]
    _agent2_r = [[] for _ in range(NENV)]
    _agent3_r = [[] for _ in range(NENV)]
    _agent1_o = [[] for _ in range(NENV)]
    _agent2_o = [[] for _ in range(NENV)]
    _agent3_o = [[] for _ in range(NENV)]

    for step in range(max_ep):
        s0 = s1
        a0 = a1
        vg0 = vg1
        vc0 = vc1
        psi0 = psi1
        phi0 = phi1
        log_logits0 = log_logits1
        was_alive = is_alive
        was_done = is_done
        cent_s0 = cent_s1

        # Run environment
        s1, reward, done, info = envs.step(actions)
        is_alive = [not va[0] for va in envs.get_avail_actions()]
        s1 = s1.astype(np.float32)  # Decentralize observation
        cent_s1 = np.expand_dims(envs.env.get_state().astype(np.float32),0)
        episode_rew += reward

        # Run decentral network
        validActions = envs.get_avail_actions()
        actions, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(s1,validActions)

        reward_pred_list.append(reward_pred1.reshape(-1))

        # Buffer
        for env_idx in range(NENV):
            for agent_id in range(num_agent):
                idx = env_idx * num_agent + agent_id
                # if not was_done[env_idx] and was_alive[idx]: # fixed length
                # Decentral trajectory
                trajs[env_idx][agent_id].append(
                    [
                        s0[idx],
                        a0[idx],
                        reward,  # + reward_pred1[idx], # Advantage
                        was_alive[idx],  # done[env_idx],masking
                        s1[idx],
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
                        cent_s0[env_idx],
                        cent_s1[env_idx],
                    ]
                )
            # Central trajectory
            cent_trajs[env_idx].append([
                cent_s0[env_idx],
                reward,
                done[env_idx],
                cent_s1[env_idx],
                ])

        if log_save_analysis:
            for env_idx in range(NENV):
                _states[env_idx].append(cent_s0[env_idx])
                _agent1_r[env_idx].append(reward_pred1[env_idx * 3])
                _agent2_r[env_idx].append(reward_pred1[env_idx * 3 + 1])
                _agent3_r[env_idx].append(reward_pred1[env_idx * 3 + 2])
                _agent1_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3]))
                _agent2_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3 + 1]))
                _agent3_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3 + 2]))
        if done.all():
            continue
    etime_roll = time.time()

    # decentralize training
    dec_batch.extend(trajs)
    if len(dec_batch) * 200 * num_agent > minimum_batch_size:
        stime_train = time.time()
        log = interval_flag(global_episodes, save_image_frequency, "im_log")
        log_image = interval_flag(global_episodes, 1024, "ima_log")
        train_decentral(
            dec_batch,
            epoch=epoch,
            batch_size=minibatch_size,
            writer=writer,
            log=log,
            step=global_episodes,
            log_image=log_image,
        )
        etime_train = time.time()
        dec_batch = []
        log_traintime.append(etime_train - stime_train)
    # centralize training
    batch.extend(cent_trajs)
    if len(batch) * 200 >= minimum_batch_size // 2:
        log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
        train_central(network, batch, 0, epoch, minibatch_size, writer, log_tc_on, global_episodes)
        batch = []

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
            tb_log_histogram(
                np.array(reward_pred_list),
                tag + "predicted_rewards",
                step=global_episodes,
            )
            writer.flush()

    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)

    # Save Gameplay
    if log_save_analysis:
        fig = plt.figure(figsize=(8, 9))
        widths = [1.2, 0.5, 1.5, 1.5]
        heights = [2, 2, 4, 4, 4]
        gs = fig.add_gridspec(
            nrows=5, ncols=4, width_ratios=widths, height_ratios=heights
        )
        ax_env = fig.add_subplot(gs[:2, :2])
        ax_env.set_title("State")
        ax_env.set_xticks([])
        ax_env.set_yticks([])
        ax_value = fig.add_subplot(gs[:2, 2:])
        ax_value.set_title("Global Value/Reward")
        ax_value.autoscale(True)
        ax_agent1 = fig.add_subplot(gs[2, 0])
        ax_agent1.set_ylabel("Agent 1")
        ax_agent1.set_xticks([])
        ax_agent1.set_yticks([])
        ax_agent2 = fig.add_subplot(gs[3, 0])
        ax_agent2.set_ylabel("Agent 2")
        ax_agent2.set_xticks([])
        ax_agent2.set_yticks([])
        ax_agent3 = fig.add_subplot(gs[4, 0])
        ax_agent3.set_ylabel("Agent 3")
        ax_agent3.set_xticks([])
        ax_agent3.set_yticks([])
        ax_reward3 = fig.add_subplot(gs[4, 1:])
        ax_reward3.autoscale(True)
        ax_reward2 = fig.add_subplot(gs[3, 1:], sharex=ax_reward3)
        ax_reward2.set_xticks([])
        ax_reward2.autoscale(True)
        ax_reward1 = fig.add_subplot(gs[2, 1:], sharex=ax_reward3)
        ax_reward1.set_xticks([])
        ax_reward1.autoscale(True)

        env_image = ax_env.imshow(np.ones((map_size, map_size, 3)), vmin=0, vmax=1)
        agent_obs1 = ax_agent1.imshow(np.ones((59, 59, 3)), vmin=0, vmax=1)
        agent_obs2 = ax_agent2.imshow(np.ones((59, 59, 3)), vmin=0, vmax=1)
        agent_obs3 = ax_agent3.imshow(np.ones((59, 59, 3)), vmin=0, vmax=1)
        env_reward_plot, env_value_plot = ax_value.plot([], [], [], [])
        (reward_plot1,) = ax_reward1.plot([], [])
        (reward_plot2,) = ax_reward2.plot([], [])
        (reward_plot3,) = ax_reward3.plot([], [])

        plt.subplots_adjust(wspace=0.33, hspace=0.33)

        fig.canvas.draw()

        def animate(i, info, critic, env_idx):
            # Environment image
            env_image.set_data(info["saved_board_rgb"][i])
            agent_obs1.set_data(_agent1_o[env_idx][i])
            agent_obs2.set_data(_agent2_o[env_idx][i])
            agent_obs3.set_data(_agent3_o[env_idx][i])

            env_reward_plot.set_data(np.arange(i), info["rewards"][:i])
            env_value_plot.set_data(np.arange(i), critic[:i])
            ax_value.relim()
            ax_value.autoscale_view()

            reward_plot1.set_data(np.arange(i), _agent1_r[env_idx][:i])
            ax_reward1.relim()
            ax_reward1.autoscale_view()
            reward_plot2.set_data(np.arange(i), _agent2_r[env_idx][:i])
            ax_reward2.relim()
            ax_reward2.autoscale_view()
            reward_plot3.set_data(np.arange(i), _agent3_r[env_idx][:i])
            ax_reward3.relim()
            ax_reward3.autoscale_view()

            return (
                ax_env,
                ax_value,
                ax_reward1,
                ax_reward2,
                ax_reward3,
                ax_agent1,
                ax_agent2,
                ax_agent3,
            )

        for idx, game_info in enumerate(history):
            states = np.array(_states[idx])
            env_critic, _ = network.run_network_central(states)
            critic = env_critic["critic"].numpy()[:, 0]

            path = os.path.join(SAVE_PATH, str(global_episodes), str(idx) + ".mp4")
            path_create(os.path.join(SAVE_PATH, str(global_episodes)))
            fg = plt.gcf()
            anim = FuncAnimation(
                fig,
                partial(animate, info=game_info, critic=critic, env_idx=idx),
                frames=len(game_info["rewards"]),
                interval=500,
            )
            anim.save(path)
            if idx > 5:
                break
        print("save animation done")
        _states = None
        _agent1_r = None
        _agent2_r = None
        _agent3_r = None
        _agent1_o = None
        _agent2_o = None
        _agent3_o = None
        anim = None

        plt.close("all")
