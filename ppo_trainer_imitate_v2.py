import pickle

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

from method.ppo2 import PPO as Network

PROGBAR = True
LOG_DEVICE = False

## Training Directory Reset
TRAIN_NAME = 'ppo_imitate_linear_v2_k3_lr5'
IMITATE_NAME = 'imitate_baseline'
LOG_PATH = './logs/'+TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
SAVE_PATH = './save/' + TRAIN_NAME
MAP_PATH = './fair_map'
GPU_CAPACITY = 0.90

NENV = multiprocessing.cpu_count()
print('Number of cpu_count : {}'.format(NENV))

env_setting_path = 'setting_full.ini'

## Import Shared Training Hyperparameters
config = configparser.ConfigParser()
config.read('config.ini')

# Training
total_episodes = 300000#config.getint('TRAINING', 'TOTAL_EPISODES')
max_ep         = config.getint('TRAINING', 'MAX_STEP')
gamma          = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd          = config.getfloat('TRAINING', 'GAE_LAMBDA')
ppo_e          = config.getfloat('TRAINING', 'PPO_EPSILON')
critic_beta    = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta   = config.getfloat('TRAINING', 'ENTROPY_BETA')
lr_a           = config.getfloat('TRAINING', 'LR_ACTOR')
lr_c           = config.getfloat('TRAINING', 'LR_CRITIC')
lr = 5e-4#1e-4
# Log Setting
save_network_frequency = config.getint('LOG', 'SAVE_NETWORK_FREQ')
save_stat_frequency    = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
save_image_frequency   = config.getint('LOG', 'SAVE_STATISTICS_FREQ')
moving_average_step    = config.getint('LOG', 'MOVING_AVERAGE_SIZE')

# Environment/Policy Settings
action_space = config.getint('DEFAULT', 'ACTION_SPACE')
vision_range = config.getint('DEFAULT', 'VISION_RANGE')
keep_frame   = config.getint('DEFAULT', 'KEEP_FRAME')
map_size     = config.getint('DEFAULT', 'MAP_SIZE')

## PPO Batch Replay Settings
minibatch_size = 256
epoch = 2
minimum_batch_size = 6000

## Setup
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 7 * keep_frame
input_size = [None, vision_dx, vision_dy, nchannel]
selfplay_reload = 20000

## Logger Initialization
log_episodic_reward = MovingAverage(moving_average_step)
log_length = MovingAverage(moving_average_step)
log_winrate = MovingAverage(moving_average_step)
log_redwinrate = MovingAverage(moving_average_step)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Map Setting
map_list = [os.path.join(MAP_PATH, path) for path in os.listdir(MAP_PATH)]
max_epsilon = 0.70;
def use_this_map():
    if np.random.random() < max_epsilon:
        return random.choice(map_list)
    else:
        return None

## Policy Setting
heur_policy_list = [policy.Patrol, policy.Roomba, policy.Defense, policy.Random, policy.AStar]
heur_weight = [1,1,1,1,1]
heur_weight = np.array(heur_weight) / sum(heur_weight)
def use_this_policy():
    return np.random.choice(heur_policy_list, p=heur_weight)

## Environment Initialization
def make_env(map_size):
    return lambda: gym.make(
            'cap-v0',
            map_size=map_size,
            config_path=env_setting_path
            )
envs = [make_env(map_size) for i in range(NENV)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue()[0])
num_red = len(envs.get_team_red()[0])

## Launch TF session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=LOG_DEVICE, allow_soft_placement=True)

if PROGBAR:
    progbar = tf.keras.utils.Progbar(None)

sess = tf.Session(config=config)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, NENV)
with tf.device('/gpu:0'):
    network = Network(input_shape=input_size, action_size=action_space, scope='main', sess=sess,lr=lr)

# Resotre / Initialize
global_episodes = 0
saver = tf.train.Saver(max_to_keep=3, var_list=network.get_vars+[global_step])
network.initiate(saver, MODEL_PATH)
global_episodes = sess.run(global_step)

writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

## Make Red's Policy
forward_network_a = TrainedNetwork(
        model_name=IMITATE_NAME,
        input_tensor='main/state:0',
        output_tensor='main/PPO/activation/Softmax:0',
        import_scope='forward',
        device='/device:GPU:0'
    )
# forward_network_v = TrainedNetwork(
#         model_name=IMITATE_NAME,
#         input_tensor='main/state:0',
#         output_tensor='main/PPO/Reshape:0',
#         import_scope='forward',
#         device='/device:GPU:0'
#     )

def prob2act(prob):
    action_out = [np.random.choice(5, p=p/sum(p)) for p in prob]
    return action_out

### TRAINING ###
def train(trajs, bootstrap=0.0, epoch=epoch, batch_size=minibatch_size, writer=None, log=False, global_episodes=None):
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
        network.update_network(*mdp_tuple, global_episodes, writer, log)

def get_action(states):
    blue_index = np.arange(len(states)).reshape((NENV,num_blue))[:,:num_blue].reshape([-1])
    # red_index = np.arange(len(states)).reshape((NENV,num_blue+num_red))[:,-num_red:].reshape([-1])
    blue_state = states[blue_index]
    # red_state = states[red_index]

    feed_dict={network.state_input: blue_state}
    ops = [network.actor, network.critic, network.log_logits]
    blue_logit, b_v1, logits1 = sess.run(ops, feed_dict)
    red_logit = forward_network_a.sess.run(forward_network_a.action,
            feed_dict={forward_network_a.state: blue_state})
    # imit_v1 = forward_network_v.sess.run(forward_network_a.action,
    #         feed_dict={forward_network_a.state: blue_state})

    blue_a = prob2act(blue_logit)
    red_a = prob2act(red_logit)

    a1 = blue_a

    actions = np.reshape(blue_a, [NENV, num_blue])
    b_logits =np.reshape(blue_logit, [NENV, num_blue,5])
    r_logits =np.reshape(red_logit, [NENV, num_blue,5])
    #Calculating Reward from different action
    mse_imit = ((b_logits - r_logits)**2).mean(axis=2)


    # Calculating Reward from a value metric
    dv_imit = 0#b_v1 - imit_v1


    return a1, b_v1, logits1, actions, blue_state, mse_imit, dv_imit

batch = []
num_batch = 0
while True:
    log_on = interval_flag(global_episodes, save_stat_frequency, 'log')
    log_image_on = interval_flag(global_episodes, save_image_frequency, 'im_log')
    save_on = interval_flag(global_episodes, save_network_frequency, 'save')
    reload_on = interval_flag(global_episodes,selfplay_reload, 'reload')
    play_save_on = interval_flag(global_episodes, 50000, 'replay_save')

    # initialize parameters
    episode_rew = np.zeros(NENV)
    prev_rew = np.zeros(NENV)
    was_alive = [True for agent in envs.get_team_blue().flat]
    was_done = [False for env in range(NENV)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*NENV)]

    #Initializing Variables for imitation learning based on episode number:
    if global_episodes < 10000:
        beta = 0.8
    if global_episodes < 50000:
        beta = 1.0 - 0.2*global_episodes/10000
    else:
        beta=0
    k = 0.3

    # Bootstrap
    s1 = envs.reset(
            config_path=env_setting_path,
            custom_board=use_this_map(),
            policy_red=use_this_policy()
        )
    a1, v1, logits1, actions, s1, mse_imit, dv_imit = get_action(s1)

    # Rollout
    stime_roll = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1

        s1, raw_reward, done, info = envs.step(actions)
        is_alive = [agent.isAlive for agent in envs.get_team_blue().flat]

        reward = (raw_reward - prev_rew - 0.01)/100.0

        if step == max_ep:
            reward[:] = -1
            done[:] = True

        episode_rew += reward

        a1, v1, logits1, actions, s1, mse_imit, dv_imit = get_action(s1)

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue().flat):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                rew = (1-beta)*reward[env_idx] - (beta)*(k*mse_imit[env_idx][idx%4])
                trajs[idx].append([s0[idx], a[idx], reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_done = done

        if np.all(done):
            break
    etime_roll = time.time()

    batch.extend(trajs)
    num_batch += sum([len(traj) for traj in trajs])
    if num_batch >= minimum_batch_size:
        stime_train = time.time()
        train(batch, 0, epoch, minibatch_size, writer, log_image_on, global_episodes)
        etime_train = time.time()
        batch = []
        num_batch = 0
        log_traintime.append(etime_train - stime_train)

    steps = []
    for env_id in range(NENV):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))

    log_episodic_reward.extend(episode_rew.tolist())
    log_length.extend(steps)
    log_winrate.extend(envs.blue_win())
    log_redwinrate.extend(envs.red_win())
    log_looptime.append(etime_roll - stime_roll)

    global_episodes += NENV
    sess.run(global_step_next)
    if PROGBAR:
        progbar.update(global_episodes)

    if log_on:
        tag = 'baseline_training/'
        record({
            tag+'length': log_length(),
            tag+'win-rate': log_winrate(),
            tag+'redwin-rate': log_redwinrate(),
            tag+'reward': log_episodic_reward(),
            tag+'rollout_time': log_looptime(),
            tag+'train_time': log_traintime(),
        }, writer, global_episodes)

    if save_on:
        network.save(saver, MODEL_PATH+'/ctf_policy.ckpt', global_episodes)

    # if reload_on:
    #     forward_network_a.reset_network_weight()

    # if play_save_on:
    #     for i in range(NENV):
    #         with open(SAVE_PATH+f'/replay{global_episodes}_{i}.pkl', 'wb') as handle:
    #             pickle.dump(info[i], handle)
