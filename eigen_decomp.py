import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys

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
from collections import defaultdict

import policy

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.multiprocessing import SubprocVecEnv
from utility.RL_Wrapper import TrainedNetwork
from utility.logger import record
from utility.gae import gae

from method.SF import Network

device_ground = '/gpu:0'
device_air = '/gpu:0'
device_t = '/gpu:0'

PROGBAR = True
LOG_DEVICE = False
OVERRIDE = False

## Training Directory Reset
TRAIN_NAME = 'SF_TRAIN_01'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.95
N=16

NENV = multiprocessing.cpu_count() // 2
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'uav_settings.ini'

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

## Import Shared Training Hyperparameters
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

# Training
total_episodes = 1000000#config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep         = config.getint('TRAINING', 'MAX_STEP')
gamma          = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd          = config.getfloat('TRAINING', 'GAE_LAMBDA')
ppo_e          = config.getfloat('TRAINING', 'PPO_EPSILON')
critic_beta    = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta   = config.getfloat('TRAINING', 'ENTROPY_BETA')
lr_a           = config.getfloat('TRAINING', 'LR_ACTOR')
lr_c           = config.getfloat('TRAINING', 'LR_CRITIC')

# Log Setting
save_network_frequency = config.getint('LOG', 'SAVE_NETWORK_FREQ')
save_stat_frequency    = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ') // 2
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = 2#config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 4096
print(minimum_batch_size)

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_explore = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
def use_fair_map():
    return random.choice(map_list)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make(
            'cap-v0',
            map_size=map_size,
            config_path=env_setting_path
            )
envs_list = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs_list, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOG_DEVICE, allow_soft_placement=True)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name='SF Training')

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
with tf.device(device_ground):
    network_sample = Network(input_shape=input_size, action_size=action_space, scope='ground', sess=sess, N=N)
with tf.device(device_air):
    network_air_sample = Network(input_shape=input_size, action_size=action_space, scope='uav', sess=sess, N=N)

# Resotre / Initialize
global_episodes = 0
saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=4)
network_sample.initiate(saver, MODEL_PATH)
if OVERRIDE:
    sess.run(tf.assign(global_step, 0)) # Reset the counter
else:
    global_episodes = sess.run(global_step)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network_sample.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes) # It save both network

def get_action_SF(states, N=5):
    states_rsh = np.reshape(states, [NENV, num_blue+num_red]+input_size[1:])
    blue_air, blue_ground, red_air, red_ground = np.split(states_rsh, [2,6,8], axis=1)
    blue_states, red_states = np.split(states_rsh, [6], axis=1)

    # BLUE GET ACTION
    blue_air = np.reshape(blue_air, [NENV*2]+input_size[1:])
    blue_ground = np.reshape(blue_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground, phi_ground1, psi_ground1 = network_sample.run_network(blue_ground)
    action_air, value_air, logit_air, phi_air1, psi_air1 = network_air_sample.run_network(blue_air)

    action_rsh = np.concatenate([action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)
    # phi = np.concatenate([phi_air.reshape([NENV,2,N]), phi_ground.reshape([NENV,4,N])], axis=1)

    # RED GET ACTION (Comment this section to make it single-side control and return blue_states)
    red_air = np.reshape(red_air, [NENV*2]+input_size[1:])
    red_ground = np.reshape(red_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground, phi_ground2, psi_ground2 = network_sample.run_network(red_ground)
    action_air, value_air, logit_air, phi_air2, psi_air2 = network_air_sample.run_network(blue_air)

    action_rsh = np.concatenate([action_rsh, action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value, value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit, logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)
    # phi = np.concatenate([phi, phi_air.reshape([NENV,2,N]), phi_ground.reshape([NENV,4,N])], axis=1)

    psi_air = np.concatenate([ psi_air1.reshape([NENV,2,N]), psi_air2.reshape([NENV,2,N])], axis=1)
    psi_ground = np.concatenate([psi_ground1.reshape([NENV,4,N]), psi_ground2.reshape([NENV,4,N])], axis=1)
    phi_air = np.concatenate([ phi_air1.reshape([NENV,2,N]), phi_air2.reshape([NENV,2,N])], axis=1)
    phi_ground = np.concatenate([phi_ground1.reshape([NENV,4,N]), phi_ground2.reshape([NENV,4,N])], axis=1)

    # RESHAPE
    action = action_rsh.reshape([-1])
    value = value.reshape([-1])
    logit = logit.reshape([NENV*12, 5])
    # phi = phi.reshape([NENV*12, N])
    # psi = psi.reshape([NENV*12, N])

    return states, action, value, logit, action_rsh, phi_air, phi_ground, psi_air, psi_ground


batch_ground, batch_air = [], []
num_batch = 0


SF_samples_air = []
SF_samples_ground = []
samples_air = 0
samples_ground = 0
#Collecting Samples for eigenvalue processing.
while (samples_ground < N or samples_air < N):
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    play_save_on = interval_flag(global_episodes, 5000, 'replay_save')

    # initialize parameters
    episode_rew = np.zeros(NENV)
    was_alive = [True for agent in range(NENV*(num_blue*num_red))]
    was_done = [False for env in range(NENV)]
    is_air = np.array([agent.is_air for agent in envs.get_team_blue().flat]).reshape([NENV, num_blue])
    is_air_red = np.array([agent.is_air for agent in envs.get_team_red().flat]).reshape([NENV, num_red])
    is_air = np.concatenate([is_air, is_air_red], axis=1).reshape([-1])

    trajs = [Trajectory(depth=8) for _ in range((num_blue+num_red)*NENV)] # Trajectory per agent

    # Bootstrap
    s1 = envs.reset(config_path=env_setting_path)
    s1, a1, v1, logits1, actions, phi_air, phi_ground, psi_air, psi_ground = get_action_SF(s1,N=N)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        # psi0 = psi
        logits = logits1

        s1, reward, done, info = envs.step(actions)

        reward_red = np.array([i['red_reward'] for i in info])
        env_reward = np.vstack((reward, reward_red)).T.reshape([-1])

        is_alive = np.array([agent.isAlive for agent in envs.get_team_blue().flat]).reshape([NENV, num_blue])
        is_alive_red = np.array([agent.isAlive for agent in envs.get_team_red().flat]).reshape([NENV, num_red])
        is_alive = np.concatenate([is_alive, is_alive_red], axis=1).reshape([-1])


        episode_rew += reward * (1-np.array(was_done, dtype=int))

        s1, a1, v1, logits1, actions, phi_air, phi_ground, psi_air, psi_ground = get_action_SF(s1,N=N)

        if samples_ground < N:
            SF_samples_ground.append(psi_ground)
        if samples_air < N:
            SF_samples_air.append(psi_air)
        samples_air+=2
        samples_ground+=2



        if np.all(done) or (samples_ground >= N and samples_air >= N):
            break

#Ground Eigenvalue Decomposition
for i,sample in enumerate(SF_samples_ground):
    if i==0:
        SF_Matrix_g=sample
    else:
        SF_Matrix_g= np.concatenate([SF_Matrix_g,sample],axis=1)
SF_Matrix_g = np.resize(SF_Matrix_g,[N,N])
w_g,v_g = np.linalg.eig(SF_Matrix_g)
##Air Eigenvalue Decompostion
for i,sample in enumerate(SF_samples_air):
    if i==0:
        SF_Matrix_a=sample
    else:

        SF_Matrix_a= np.concatenate([SF_Matrix_a,sample],axis=1)
SF_Matrix_a = np.resize(SF_Matrix_a,[N,N])
w_a,v_a = np.linalg.eig(SF_Matrix_a)


TRAIN_NAME = 'DECOMP_TRAIN_01'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME

#Choose the Max three Eigenvalues to continue with and their corresponding Eigenvectors.
#The reward for each of the subpolicies is the vector multiplied with the Phi output.
num_mode = 5
eigValIdx = w_g.argsort()[-num_mode:][::-1]
eigVect_g = []
for idx in eigValIdx:
    eigVect_g.append(v_g[:,idx])

eigValIdx = w_a.argsort()[-num_mode:][::-1]
eigVect_a = []
for idx in eigValIdx:
    eigVect_a.append(v_a[:,idx])

##Creating the new subpolicies based on the eigen decomposition.
from method.ppo import PPO_multimodes as Network2

heur_policy_list = [policy.Patrol, policy.Roomba, policy.Defense, policy.Random, policy.AStar]
heur_weight = [1,1,1,1,1]
heur_weight = np.array(heur_weight) / sum(heur_weight)
def use_this_policy():
    return np.random.choice(heur_policy_list, p=heur_weight)

with tf.device(device_t):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step, NENV)
    subtrain_step = [tf.Variable(0, trainable=False) for _ in range(num_mode)]
    subtrain_step_next = [tf.assign_add(step, NENV) for step in subtrain_step]
    network_g = Network2(in_size=input_size, action_size=action_space, scope='main_ground', sess=sess, num_mode=num_mode, model_path=MODEL_PATH+"ground")
    network_a = Network2(in_size=input_size, action_size=action_space, scope='main_air', sess=sess, num_mode=num_mode, model_path=MODEL_PATH+"air")

network_g.initiate(saver, MODEL_PATH)
network_a.initiate(saver, MODEL_PATH)
network_g.save(saver, MODEL_PATH+"ground"+'/ctf_policy.ckpt', global_episodes)
network_a.save(saver, MODEL_PATH+"air"+'/ctf_policy.ckpt', global_episodes)


#Reward Shaping based on the Eigen Decomposition.
def reward_shape(phi, air, idx=None, additional_reward=None):
    if air:
        eigVect = eigVect_a
    else:
        eigVect = eigVect_g

    reward = []
    for i in range(NENV):
        rew = 0
        for j in range(phi.shape[1]):
            rew += np.sum(eigVect[idx]*phi[i,j,:])
        reward.append(rew)

    if additional_reward is not None:
        return np.array(reward) + additional_reward
    else:
        return np.array(reward)

print('Training Initiated:')
def get_action(states):
    states_rsh = np.reshape(states, [NENV, num_blue+num_red]+input_size[1:])
    blue_air, blue_ground, red_air, red_ground = np.split(states_rsh, [2,6,8], axis=1)
    blue_states, red_states = np.split(states_rsh, [6], axis=1)

    # BLUE GET ACTION
    blue_air = np.reshape(blue_air, [NENV*2]+input_size[1:])
    blue_ground = np.reshape(blue_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground = network_g.run_network(blue_ground, MODE)
    action_air, value_air, logit_air = network_a.run_network(blue_air, MODE)

    action_rsh = np.concatenate([action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)

    # RED GET ACTION (Comment this section to make it single-side control and return blue_states)
    red_air = np.reshape(red_air, [NENV*2]+input_size[1:])
    red_ground = np.reshape(red_ground, [NENV*4]+input_size[1:])

    action_ground, value_ground, logit_ground = network_g.run_network(red_ground, MODE)
    action_air, value_air, logit_air = network_a.run_network(blue_air, MODE)

    action_rsh = np.concatenate([action_rsh, action_air.reshape([NENV,2]), action_ground.reshape([NENV,4])], axis=1)
    value = np.concatenate([value, value_air.reshape([NENV,2]), value_ground.reshape([NENV,4])], axis=1)
    logit = np.concatenate([logit, logit_air.reshape([NENV,2,5]), logit_ground.reshape([NENV,4,5])], axis=1)

    # RESHAPE
    action = action_rsh.reshape([-1])
    value = value.reshape([-1])
    logit = logit.reshape([NENV*12, 5])

    return states, action, value, logit, action_rsh

def train(trajs, updater, bootstrap=0, epoch=epoch, batch_size=minibatch_size, **kwargv):
    traj_buffer = defaultdict(list)
    buffer_size = 0
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_size += len(traj)

        td_target, advantages = gae(traj[2], traj[3], 0,
                gamma, lambd, normalize=False)

        traj_buffer['state'].extend(traj[0])
        traj_buffer['action'].extend(traj[1])
        traj_buffer['td_target'].extend(td_target)
        traj_buffer['advantage'].extend(advantages)
        traj_buffer['logit'].extend(traj[4])

    if buffer_size < 10:
        return

    it = batch_sampler(
            batch_size,
            epoch,
            np.stack(traj_buffer['state']),
            np.stack(traj_buffer['action']),
            np.stack(traj_buffer['td_target']),
            np.stack(traj_buffer['advantage']),
            np.stack(traj_buffer['logit'])
        )
    for mdp_tuple in it:
        updater(*mdp_tuple, **kwargv, idx=MODE)

batch = []
num_batch = 0
mode_changed = False
MODE = 0
config_path = "setting_partial.ini"
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)
while True:
    if mode_changed:
        mode_changed = False
        MODE = (MODE + 1) % num_mode
    #MODE = np.argmin(sess.run(subtrain_step))
    mode_episode = sess.run(subtrain_step[MODE])

    log_on = interval_flag(mode_episode, save_stat_frequency, 'log{}'.format(MODE))
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    reload_on = False # interval_flag(global_episodes,selfplay_reload, 'reload')

    # Bootstrap
    # s1 = envs.reset(
    #         config_path=config_path,
    #         custom_board=use_fair_map(),
    #         policy_red=use_this_policy()
    #     )
    s1 = envs.reset(config_path=env_setting_path)
    num_blue = len(envs.get_team_blue()[0])
    num_red = len(envs.get_team_red()[0])

    # initialize parameters
    episode_rew = np.zeros(NENV)
    episode_env_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in range(NENV*(num_blue*num_red))]
    was_alive_red = [True for agent in envs.get_team_red().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range((num_blue+num_red)*NENV)]


    s1, a1, v1, logits1, actions = get_action(s1)

    _,_,_,_,_, phi_air, phi_ground, _, _ = get_action_SF(s1,N=N)

    # Rollout
    stime = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = np.array([agent.isAlive for agent in envs.get_team_blue().flat]).reshape([NENV, num_blue])
        is_alive_red = np.array([agent.isAlive for agent in envs.get_team_red().flat]).reshape([NENV, num_red])
        is_alive = np.concatenate([is_alive, is_alive_red], axis=1).reshape([-1])

        if step == max_ep:
            done[:] = True
        reward_a = reward_shape(phi_air, True, MODE)
        reward_g = reward_shape(phi_ground, False, MODE)

        if step == max_ep:
            env_reward[:] = -1
        else:
            env_reward = (raw_reward - prev_rew - 0.01)/100


        for i in range(NENV):
            if not was_done[i]:
                episode_rew[i] += reward[i]
                episode_env_rew[i] += env_reward[i]

        s1, a1, v1, logits1, actions = get_action(s1)
        _,_,_,_,_, phi_air, phi_ground, psi_air, psi_ground = get_action_SF(s1,N=N)

        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx in range(NENV*(num_blue+num_red)):
            env_idx = idx // (num_blue+num_red)
            env_team_idx = idx // 6
            if was_alive[idx] and not was_done[env_idx]:
                if is_air[idx]:
                    #agent_reward = air_reward[env_team_idx] + env_reward[env_team_idx]
                    agent_reward = reward_a[env_idx]
                else:
                    agent_reward = reward_g[env_idx]
                trajs[idx].append([s0[idx], a[idx], agent_reward, v0[idx], logits[idx],])
        # Split air trajectory and ground trajectory

        prev_rew = raw_reward
        was_alive = is_alive
        was_alive_red = is_alive_red
        was_done = done

        if np.all(done):
            break
    etime_roll = time.time()

    global_episodes += NENV
    sess.run(global_step_next)
    sess.run(subtrain_step_next[MODE])
    if PROGBAR:
        progbar.update(global_episodes)

    for idx in range(NENV*(num_blue+num_red)):
        if is_air[idx]:
            batch_air.append(trajs[idx])
        else:
            batch_ground.append(trajs[idx])
    num_batch += sum([len(traj) for traj in trajs])

    if num_batch >= minimum_batch_size:
        train(batch_air, network_a.update_global, 0, epoch, minibatch_size, writer=writer, log=log_image_on, global_episodes=global_episodes)
        train(batch_ground, network_g.update_global, 0, epoch, minibatch_size, writer=writer, log=log_image_on, global_episodes=global_episodes)
        batch = []
        num_batch = 0
        mode_changed = True

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))


    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime)

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        tag = 'uav_training/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'redwin-rate': log_redwinrate(),
            tag+'env_reward': log_episodic_reward(),
            tag+'rollout_time': log_looptime(),
        }, writer, global_episodes)

    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)
