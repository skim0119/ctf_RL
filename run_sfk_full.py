import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import sys

import shutil
import configparser
import argparse

import signal
import threading
import multiprocessing

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
#from utility.slack import SlackAssist

from method.dist import SF_CVDC as Network

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'DIST_SF_CVDC_22'
TRAIN_TAG = 'Central value decentralized control, '+TRAIN_NAME
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_3g_20'
GPU_CAPACITY = 0.95

#slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")

NENV = multiprocessing.cpu_count()
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'env_setting_3v3_3g_full.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

## Import Shared Training Hyperparameters
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep         = 200#config.getint('TRAINING', 'MAX_STEP')
gamma          = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd          = config.getfloat('TRAINING', 'GAE_LAMBDA')
ppo_e          = config.getfloat('TRAINING', 'PPO_EPSILON')
critic_beta    = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta   = config.getfloat('TRAINING', 'ENTROPY_BETA')
lr_a           = config.getfloat('TRAINING', 'LR_ACTOR')
lr_c           = config.getfloat('TRAINING', 'LR_CRITIC')

# Log Setting
save_network_frequency = config.getint('LOG', 'SAVE_NETWORK_FREQ')
save_stat_frequency    = 128#config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = 128#config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 1#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 1024 * 2
print(minimum_batch_size)

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]
cent_input_size = [None, map_size, map_size, nchannel]

## Logger Initialization 
log_episodic_reward = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Environment Initialization
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH) if path[:5]=='board']
_qenv = gym.make('cap-v0', map_size=map_size, config_path=env_setting_path)
def make_env(map_size):
    return lambda: gym.make('cap-v0', map_size=map_size,
                            config_path=env_setting_path)
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame=keep_frame, size=vision_dx)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])
num_agent = num_blue#+num_red

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

atoms = 64
network = Network(
        central_obs_shape=cent_input_size,
        decentral_obs_shape=input_size,
        action_size=action_space, 
        atoms=atoms,
        save_path=MODEL_PATH)

# Resotre / Initialize
global_episodes = network.initiate()
print(global_episodes)
input('start?')

writer = tf.summary.create_file_writer(LOG_PATH)
#network.save(global_episodes)

### TRAINING ###
def make_ds(features, labels, shuffle_buffer_size=64, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(shuffle_buffer_size, seed=0)
    if repeat:
        ds = ds.repeat()
    return ds

def train_central(network, trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue

        # Forward
        states = np.array(traj[0])
        last_state = np.array(traj[3])[-1:,:,:,:]
        env_critic, _ = network.run_network_central(states) 
        _env_critic, _ = network.run_network_central(last_state)
        critic = env_critic['critic'].numpy()[:,0].tolist()
        _critic = _env_critic['critic'].numpy()[0,0]
        phi = env_critic['phi'].numpy().tolist()
        psi = env_critic['psi'].numpy().tolist()
        _psi = _env_critic['psi'].numpy()[0]

        reward = traj[1]
        #critic = traj[6]
        #phi = np.array(traj[7]).tolist()
        #psi = np.array(traj[8]).tolist()
        
        td_target_c, advantages = gae(reward, critic, _critic,
                gamma, lambd, normalize=False)
        td_target, _ = gae(phi, psi, _psi,
                gamma, lambd, normalize=False)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['reward'].extend(traj[1])
        traj_buffer['done'].extend(traj[2])
        traj_buffer['next_state'].extend(traj[3])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['td_target_c'].extend(td_target_c)

    train_dataset = tf.data.Dataset.from_tensor_slices({
            'state': np.stack(traj_buffer['state']),
            'td_target': np.stack(traj_buffer['td_target']),
            'td_target_c': np.stack(traj_buffer['td_target_c']),
        }).shuffle(64).repeat(epoch).batch(batch_size)
    
    psi_losses = []
    elbo_losses = []
    critic_losses = []
    for inputs in train_dataset:
        info = network.update_central_critic(inputs)
        if log:
            psi_losses.append(info['psi_mse'])
            elbo_losses.append(info['elbo'])
            critic_losses.append(info['critic_mse'])
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_psi_loss', np.mean(psi_losses), step=step)
            tf.summary.scalar(tag+'main_ELBO_loss', np.mean(elbo_losses), step=step)
            tf.summary.scalar(tag+'main_critic_loss', np.mean(critic_losses), step=step)
            tb_log_ctf_frame(info['sample_generated_image'], 'sample generated image', step)
            tb_log_ctf_frame(info['sample_decoded_image'], 'sample decoded image', step)
            writer.flush()

def train_decentral(agent_trajs, team_trajs, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, step=None):
    advantage_list = []
    # Agent trajectory processing
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(agent_trajs):
        if len(traj) == 0:
            continue

        reward = traj[2]
        critic = traj[5]
        phi = np.array(traj[7]).tolist()
        psi = np.array(traj[8]).tolist()
        _critic = traj[9][-1]
        _psi = np.array(traj[10][-1])
        
        # Zero bootstrap because all trajectory terminates
        td_target_c, advantages = gae(reward, critic, _critic,
                gamma, lambd, normalize=False)
        td_target, _ = gae(phi, psi, _psi,#np.zeros_like(phi[0]),
                gamma, lambd, normalize=False)
        advantage_list.append(advantages)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['log_logit'].extend(traj[6])
        traj_buffer['action'].extend(traj[1])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)
        traj_buffer['td_target_c'].extend(td_target_c)
    agent_dataset = tf.data.Dataset.from_tensor_slices({
            'state': np.stack(traj_buffer['state']),
            'old_log_logit': np.stack(traj_buffer['log_logit']),
            'action': np.stack(traj_buffer['action']),
            'td_target': np.stack(traj_buffer['td_target']),
            'advantage': np.stack(traj_buffer['advantage']),
            'td_target_c': np.stack(traj_buffer['td_target_c']),
        }).shuffle(64).repeat(epoch).batch(batch_size)

    # Team trajectory preproessing
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(team_trajs):
        if len(traj) == 0:
            continue
        reward = traj[2]
        critic = traj[3]
        _critic = traj[4][-1]
        td_target_c, _= gae(reward, critic, _critic,
                gamma, lambd, normalize=False)
        traj_buffer['team_state'].extend(traj[0])
        traj_buffer['value_central'].extend(td_target_c)
        traj_buffer['mask'].extend(traj[1])
        traj_buffer['rewards'].extend(traj[2])
    team_dataset = tf.data.Dataset.from_tensor_slices({
            'team_state': np.stack(traj_buffer['team_state']),
            'value_central': np.stack(traj_buffer['value_central']),
            'mask': np.stack(traj_buffer['mask']),
            'rewards': np.stack(traj_buffer['rewards']),
            }).shuffle(64).repeat(epoch).batch(batch_size)

    logs =  network.update_decentral(agent_dataset, team_dataset, log=log)

    if log:
        with writer.as_default():
            tag = 'summary/'
            tb_log_histogram(np.array(advantage_list), tag+'dec_advantages', step=global_episodes)
            for name, val in logs.items():
                tf.summary.scalar(tag+name, val, step=step)
            writer.flush()

def train_reward_prediction(network, traj, epoch, batch_size, writer=None, log=False, step=None):
    buffer_size = len(traj)
    if buffer_size < 10:
        return
    states = np.stack(traj[0])
    rewards = np.stack(traj[1])
    # Oversampling
    bool_pos_reward = rewards > 0
    bool_neg_reward = rewards < 0
    bool_zero_reward = rewards == 0
    pos_ds = make_ds(states[bool_pos_reward], rewards[bool_pos_reward])
    neg_ds = make_ds(states[bool_neg_reward], rewards[bool_neg_reward])
    zero_ds = make_ds(states[bool_zero_reward], rewards[bool_zero_reward])
    states = []
    rewards = []
    train_dataset = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds, zero_ds], weights=[0.1, 0.1, 0.8]).batch(128)

    reward_losses = []
    rp_losses = []
    counter = 0; max_count = 256
    for state, reward in train_dataset:
        inputs = {'state': state, 'reward': reward}
        info = network.update_reward_prediction(inputs)
        reward_losses.append(info['reward_mse'])
        rp_losses.append(info['rp_loss'])
        counter += 1
        if counter > max_count:
            break
    if log:
        with writer.as_default():
            tag = 'summary/'
            tf.summary.scalar(tag+'main_reward_loss', np.mean(reward_losses), step=step)
            tf.summary.scalar(tag+'main_rp_loss', np.mean(rp_losses), step=step)
            writer.flush()


def get_action(log_logits):
    a = tf.random.categorical(log_logits, 1, dtype=tf.int32).numpy().ravel()
    action = np.reshape(a, [NENV, num_agent])
    return a, action

reward_training_buffer = Trajectory(depth=4) # Centralized
batch = []
dec_batch = []
ma_batch = []
num_batch = 0
dec_batch_size = 0
#while global_episodes < total_episodes:
while True:
    # Flags
    log_save_analysis = interval_flag(global_episodes, 1024*4, 'save_log')

    # initialize parameters 
    episode_rew = np.zeros(NENV)
    is_alive = [True for agent in envs.get_team_blue().flat]
    is_done = [False for env in range(NENV*num_agent)]

    trajs = [Trajectory(depth=11) for _ in range(NENV*num_agent)]
    ma_trajs = [Trajectory(depth=5) for _ in range(NENV)]
    cent_trajs = [Trajectory(depth=4) for _ in range(NENV)]
    
    # Bootstrap
    if log_save_analysis:
        s1 = envs.reset(
                map_pool=map_list,
                config_path='env_setting_3v3_3g_full_rgb.ini',
                policy_red=policy.Roomba,
                #policy_blue=policy.Roomba,
                )
    else:
        s1 = envs.reset(
                map_pool=map_list,
                config_path=env_setting_path,
                policy_red=policy.Roomba,
                #policy_blue=policy.Roomba,
                )
    s1 = s1.astype(np.float32)
    cent_s1 = envs.get_obs_blue().astype(np.float32) # Centralized

    actor, critic = network.run_network_decentral(s1)
    a1, action = get_action(actor['log_softmax'].numpy())
    v1 = critic['critic'].numpy()[:,0]
    psi1 = critic['psi'].numpy()
    phi1 = critic['phi'].numpy()

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
        v0 = v1
        psi0 = psi1
        phi0 = phi1
        log_logits0 = actor['log_softmax'].numpy()
        was_alive = is_alive
        was_done = is_done
        cent_s0 = cent_s1
        
        # Run environment
        s1, reward, done, history = envs.step(action)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]
        s1 = s1.astype(np.float32) # Decentralize observation
        cent_s1 = envs.get_obs_blue().astype(np.float32) # Centralized
        episode_rew += reward

        # Run decentral network
        actor, critic = network.run_network_decentral(s1)
        a1, action = get_action(actor['log_softmax'])
        phi1 = critic['phi'].numpy()
        reward_pred1 = critic['reward_predict'].numpy()[:,0]
        v1 = critic['critic'].numpy()[:,0]
        psi1 = critic['psi'].numpy()

        reward_pred_list.append(reward_pred1.reshape(-1))

        # Buffer
        for idx in range(NENV*num_agent):
            env_idx = idx // num_agent
            #if not was_done[env_idx] and was_alive[idx]:
            if not was_done[env_idx]:
                # Decentral trajectory
                trajs[idx].append([
                    s0[idx],
                    a0[idx],
                    reward[env_idx],# + 0.01*reward_pred1[idx], # Advantage
                    done[env_idx],
                    s1[idx],
                    v0[idx], # Advantage
                    log_logits0[idx], # PPO
                    phi0[idx], # phi: one-step ahead
                    psi0[idx],
                    v1[idx],
                    psi1[idx],
                    ])
        for env_idx in range(NENV):
            if not was_done[env_idx]:
                # Reward training buffer
                reward_training_buffer.append([
                    cent_s0[env_idx],
                    reward[env_idx], ])
                # Central trajectory
                cent_trajs[env_idx].append([
                    cent_s0[env_idx],
                    reward[env_idx],
                    done[env_idx],
                    cent_s1[env_idx],
                    ])
                # MA trajectory
                ma_trajs[env_idx].append([
                    s0[env_idx*num_agent:(env_idx+1)*num_agent],
                    is_alive[env_idx*num_agent:(env_idx+1)*num_agent],
                    reward[env_idx],
                    np.sum(v0[env_idx*num_agent:(env_idx+1)*num_agent]),
                    np.sum(v1[env_idx*num_agent:(env_idx+1)*num_agent]),
                    ])

        if log_save_analysis:
            for env_idx in range(NENV):
                _states[env_idx].append(cent_s0[env_idx])
                _agent1_r[env_idx].append(reward_pred1[env_idx*3])
                _agent2_r[env_idx].append(reward_pred1[env_idx*3+1])
                _agent3_r[env_idx].append(reward_pred1[env_idx*3+2])
                _agent1_o[env_idx].append(_qenv._env2rgb(s0[env_idx*3]))
                _agent2_o[env_idx].append(_qenv._env2rgb(s0[env_idx*3+1]))
                _agent3_o[env_idx].append(_qenv._env2rgb(s0[env_idx*3+2]))

        if np.all(done):
            break
    etime_roll = time.time()

    # decentralize training
    dec_batch.extend(trajs)
    ma_batch.extend(ma_trajs)
    dec_batch_size += sum([len(traj) for traj in trajs])
    if dec_batch_size > minimum_batch_size:
        stime_train = time.time()
        log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
        train_decentral(dec_batch, ma_batch, 
                epoch=epoch, batch_size=minibatch_size,
                writer=writer, log=log_image_on, step=global_episodes)
        etime_train = time.time()
        dec_batch = []
        ma_batch = []
        dec_batch_size = 0
        log_traintime.append(etime_train - stime_train)
    # centralize training
    batch.extend(cent_trajs)
    num_batch += sum([len(traj) for traj in cent_trajs])
    if num_batch >= minimum_batch_size:
        log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
        train_central(network, batch, 0, epoch, minibatch_size, writer, log_tc_on, global_episodes)
        num_batch = 0
        batch = []
    elif len(reward_training_buffer) > 50000:
        log_rt_on = interval_flag(global_episodes, save_image_frequency, 'rt_log')
        train_reward_prediction(network, reward_training_buffer, epoch=2, batch_size=64, writer=writer, log=log_rt_on, step=global_episodes)
        reward_training_buffer.clear()
    
    log_episodic_reward.extend(episode_rew.tolist())
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    if PROGBAR:
        progbar.update(global_episodes)

    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    if log_on:
        with writer.as_default():
            tag = 'baseline_training/'
            tf.summary.scalar(tag+'win-rate', log_winrate(), step=global_episodes)
            tf.summary.scalar(tag+'redwin-rate', log_redwinrate(), step=global_episodes)
            tf.summary.scalar(tag+'env_reward', log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag+'rollout_time', log_looptime(), step=global_episodes)
            tf.summary.scalar(tag+'train_time', log_traintime(), step=global_episodes)
            tb_log_histogram(np.array(reward_pred_list), tag+'predicted_rewards', step=global_episodes)
            writer.flush()
        
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    if save_on:
        network.save(global_episodes)

    # Save Gameplay
    if log_save_analysis:
        fig = plt.figure(figsize=(8,9))
        widths = [1.2,0.5,1.5,1.5]
        heights = [2,2,4,4,4]
        gs = fig.add_gridspec(nrows=5, ncols=4,
                width_ratios=widths, height_ratios=heights)
        ax_env = fig.add_subplot(gs[:2,:2])
        ax_env.set_title('State')
        ax_env.set_xticks([])
        ax_env.set_yticks([])
        ax_value = fig.add_subplot(gs[:2, 2:])
        ax_value.set_title('Global Value/Reward')
        ax_value.autoscale(True)
        ax_agent1 = fig.add_subplot(gs[2,0])
        ax_agent1.set_ylabel('Agent 1')
        ax_agent1.set_xticks([])
        ax_agent1.set_yticks([])
        ax_agent2 = fig.add_subplot(gs[3,0])
        ax_agent2.set_ylabel('Agent 2')
        ax_agent2.set_xticks([])
        ax_agent2.set_yticks([])
        ax_agent3 = fig.add_subplot(gs[4,0])
        ax_agent3.set_ylabel('Agent 3')
        ax_agent3.set_xticks([])
        ax_agent3.set_yticks([])
        ax_reward3 = fig.add_subplot(gs[4,1:])
        ax_reward3.autoscale(True)
        ax_reward2 = fig.add_subplot(gs[3,1:], sharex=ax_reward3)
        ax_reward2.set_xticks([])
        ax_reward2.autoscale(True)
        ax_reward1 = fig.add_subplot(gs[2,1:], sharex=ax_reward3)
        ax_reward1.set_xticks([])
        ax_reward1.autoscale(True)

        env_image = ax_env.imshow(np.ones((map_size, map_size, 3)), vmin=0, vmax=1)
        agent_obs1 = ax_agent1.imshow(np.ones((39,39,3)), vmin=0, vmax=1)
        agent_obs2 = ax_agent2.imshow(np.ones((39,39,3)), vmin=0, vmax=1)
        agent_obs3 = ax_agent3.imshow(np.ones((39,39,3)), vmin=0, vmax=1)
        env_reward_plot, env_value_plot = ax_value.plot([],[], [],[])
        reward_plot1, = ax_reward1.plot([],[]) 
        reward_plot2, = ax_reward2.plot([],[]) 
        reward_plot3, = ax_reward3.plot([],[]) 

        plt.subplots_adjust(wspace=0.33, hspace=0.33)

        fig.canvas.draw()
        
        def animate(i, info, critic, env_idx):
            # Environment image
            env_image.set_data(info['saved_board_rgb'][i])
            agent_obs1.set_data(_agent1_o[env_idx][i])
            agent_obs2.set_data(_agent2_o[env_idx][i])
            agent_obs3.set_data(_agent3_o[env_idx][i])

            env_reward_plot.set_data(np.arange(i), info['rewards'][:i])
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

            return ax_env, ax_value, ax_reward1, ax_reward2, ax_reward3, ax_agent1, ax_agent2, ax_agent3

        for idx, game_info in enumerate(history):
            states = np.array(_states[idx])
            env_critic, _ = network.run_network_central(states) 
            critic = env_critic['critic'].numpy()[:,0]

            path = os.path.join(SAVE_PATH, str(global_episodes), str(idx)+'.mp4')
            path_create(os.path.join(SAVE_PATH, str(global_episodes)))
            fg = plt.gcf()
            anim = FuncAnimation(fig, partial(animate, info=game_info, critic=critic, env_idx=idx),
                    frames=len(game_info['rewards']), interval=500)
            anim.save(path)
            if idx > 5:
                break
        print('save animation done')
        _states = None
        _agent1_r = None
        _agent2_r = None
        _agent3_r = None
        _agent1_o = None
        _agent2_o = None
        _agent3_o = None
        anim = None

        plt.close('all')

