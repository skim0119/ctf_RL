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

from method.dist import SF_CVDC as Network

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
TRAIN_NAME = "CVDC_{}_{:02d}_convoy_{}g{}a_{}g{}a_m{}".format(
    args.machine,
    args.train_number,
    args.nbg,
    args.nba,
    args.nrg,
    args.nra,
    args.map_size,
)
TRAIN_TAG = "Central value decentralized control(learnability), " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
SAVE_PATH = "./save/" + TRAIN_NAME
MAP_PATH = "./fair_3g_20"
GPU_CAPACITY = 0.95

# slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")

NENV = multiprocessing.cpu_count() // 2
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
path_create(SAVE_PATH)

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
keep_frame = 1
map_size = args.map_size
vision_range = map_size - 1
vision_dx, vision_dy = 2 * vision_range + 1, 2 * vision_range + 1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]
cent_input_size = [None, map_size, map_size, nchannel]
## Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 1024 * 4

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
        if len(traj) == 0:
            continue

        # Forward
        states = np.array(traj[0])
        last_state = np.array(traj[3])[-1:, :, :, :]
        env_critic, _ = network.run_network_central(states)
        _env_critic, _ = network.run_network_central(last_state)
        critic = env_critic["critic"].numpy()[:, 0].tolist()
        _critic = _env_critic["critic"].numpy()[0, 0]
        phi = env_critic["phi"].numpy().tolist()
        psi = env_critic["psi"].numpy().tolist()
        _psi = _env_critic["psi"].numpy()[0]

        reward = traj[1]

        td_target_c, advantages = gae(reward, critic, _critic, gamma, lambd)
        td_target_psi, _ = gae(phi, psi, _psi, gamma, lambd, normalize=False)

        traj_buffer["state"].extend(traj[0])
        traj_buffer["reward"].extend(traj[1])
        traj_buffer["done"].extend(traj[2])
        traj_buffer["next_state"].extend(traj[3])
        traj_buffer["td_target_psi"].extend(td_target_psi)
        traj_buffer["td_target_c"].extend(td_target_c)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]),
                "td_target": np.stack(traj_buffer["td_target_psi"]),
                "td_target_c": np.stack(traj_buffer["td_target_c"]),
            }
        )
        .shuffle(64)
        .repeat(epoch)
        .batch(batch_size)
    )

    psi_losses = []
    elbo_losses = []
    critic_losses = []
    for inputs in train_dataset:
        info = network.update_central_critic(inputs)
        if log:
            psi_losses.append(info["psi_mse"])
            elbo_losses.append(info["elbo"])
            critic_losses.append(info["critic_mse"])
    if log:
        with writer.as_default():
            tag = "summary/"
            tf.summary.scalar(tag + "main_psi_loss", np.mean(psi_losses), step=step)
            tf.summary.scalar(tag + "main_ELBO_loss", np.mean(elbo_losses), step=step)
            tf.summary.scalar(
                tag + "main_critic_loss", np.mean(critic_losses), step=step
            )
            tb_log_ctf_frame(
                info["sample_generated_image"], "sample generated image", step
            )
            tb_log_ctf_frame(info["sample_decoded_image"], "sample decoded image", step)
            writer.flush()


def train_decentral(
    agent_trajs,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    train_datasets = []

    # Agent trajectory processing
    traj_buffer_list = [defaultdict(list) for _ in range(num_type)]
    advantage_lists = [[] for _ in range(num_type)]
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
            td_target_psi, _ = gae(
                phi,
                psi,
                _psi,  # np.zeros_like(phi[0]),
                gamma,
                lambd,
               # mask=np.array(mask)[:, None],
                normalize=False,
            )
            advantage_lists[atype].append(advantages)

            traj_buffer = traj_buffer_list[atype]
            traj_buffer["state"].extend(traj[0])
            traj_buffer["next_state"].extend(traj[4])
            traj_buffer["log_logit"].extend(traj[6])
            traj_buffer["action"].extend(traj[1])
            traj_buffer["old_value"].extend(critic)
            traj_buffer["td_target_psi"].extend(td_target_psi)
            #traj_buffer["advantage"].extend(advantages)
            traj_buffer["advantage"].extend(advantages_global)
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
        train_datasets, writer=writer, log=log, step=step, tag="losses/"
    )
    if log:
        with writer.as_default():
            tag = "advantages/"
            for idx, adv_list in enumerate(advantage_lists):
                tb_log_histogram(
                    np.array(adv_list), tag + "dec_advantages_{}".format(idx), step=step
                )
            writer.flush()


def run_network(states):
    # State Process
    states_list = []
    for mask in agent_type_masking:
        state = np.compress(mask, states, axis=0)
        states_list.append(state)

    # Run network
    results = network.run_network_decentral(states_list)

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
    action = np.reshape(a1, [NENV, num_blue])
    return action, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1


batch = []
dec_batch = []
num_batch = 0
dec_batch_size = 0
#while global_episodes < total_episodes:
while True:
    # Flags
    log_save_analysis = False  #interval_flag(global_episodes, 1024 * 4, "save_log")

    # initialize parameters
    episode_rew = np.zeros(NENV)
    is_alive = [True for agent in envs.get_team_blue().flat]
    is_done = [False for env in range(NENV * num_agent)]

    trajs = [[Trajectory(depth=14) for _ in range(num_agent)] for _ in range(NENV)]
    cent_trajs = [Trajectory(depth=4) for _ in range(NENV)]

    # Bootstrap
    if log_save_analysis:
        game_config["experiments"]["SAVE_BOARD_RGB"] = "True"
    else:
        game_config["experiments"]["SAVE_BOARD_RGB"] = "False"
    s1 = envs.reset(config_path=game_config, policy_red=policy.Roomba,)
    s1 = s1.astype(np.float32)
    #cent_s1 = envs.get_obs_blue().astype(np.float32)  # Centralized

    actions, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(s1)

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
        #cent_s0 = cent_s1

        # Run environment
        s1, reward, done, history = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        s1 = s1.astype(np.float32)  # Decentralize observation
        #cent_s1 = envs.get_obs_blue().astype(np.float32)  # Centralized
        episode_rew += reward

        # Run decentral network
        actions, a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1 = run_network(s1)

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
                        reward[env_idx],  # + reward_pred1[idx], # Advantage
                        was_alive[idx],  # done[env_idx],masking
                        s1[idx],
                        vg0[idx],  # Advantage
                        log_logits0[idx],  # PPO
                        phi0[idx],  # phi: one-step ahead
                        psi0[idx],
                        vg1[idx],
                        psi1[idx],
                        reward[env_idx],  #  - (reward_pred1[idx] if reward[env_idx] else 0),
                        vc0[idx],
                        vc1[idx],
                    ]
                )
            # Central trajectory
            """
            cent_trajs[env_idx].append([
                cent_s0[env_idx],
                reward[env_idx],
                done[env_idx],
                cent_s1[env_idx],
                ])
            """

        if log_save_analysis:
            for env_idx in range(NENV):
                _states[env_idx].append(cent_s0[env_idx])
                _agent1_r[env_idx].append(reward_pred1[env_idx * 3])
                _agent2_r[env_idx].append(reward_pred1[env_idx * 3 + 1])
                _agent3_r[env_idx].append(reward_pred1[env_idx * 3 + 2])
                _agent1_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3]))
                _agent2_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3 + 1]))
                _agent3_o[env_idx].append(_qenv._env2rgb(s0[env_idx * 3 + 2]))
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
        log_traintime.append(etime_train - stime_train)
    # centralize training
    """
    batch.extend(cent_trajs)
    num_batch += sum([len(traj) for traj in cent_trajs])
    if num_batch >= minimum_batch_size:
        log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
        train_central(network, batch, 0, epoch, minibatch_size, writer, log_tc_on, global_episodes)
        num_batch = 0
        batch = []
    """

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
