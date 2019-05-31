import os
import shutil
import configparser

import signal
import threading
import multiprocessing

import tensorflow as tf

import time
import gym
import gym_cap
import gym_cap.envs.const as CONST
import numpy as np
import random
import math

# the modules that you can use to generate the policy. 
import policy.random
import policy.roomba
import policy.zeros

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MovingAve
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory

from network.a3c import ActorCritic as AC

from network.base import initialize_uninitialized_vars as iuv

OVERRIDE = True;
TRAIN_NAME='ActorCritic'
LOG_PATH='./logs/'+TRAIN_NAME
MODEL_PATH='./model/' + TRAIN_NAME
GPU_CAPACITY=0.5 # gpu capacity in percentage

if OVERRIDE:
    #  Remove and reset log and model directory
    #  !rm -rf logs/A3C_benchmark/ model/A3C_benchmark
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH,ignore_errors=True)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH,ignore_errors=True)

# Create model and log directory
if not os.path.exists(MODEL_PATH):
    try:
        os.makedirs(MODEL_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {MODEL_PATH} failed')
if not os.path.exists(LOG_PATH):
    try:
        os.makedirs(LOG_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {LOG_PATH} failed')

## Hyperparameters
## Environment
action_space = 5 
num_blue = 4
map_size = 20
vision_range = 19 

## Training
total_episodes = 1e6
step_limit = 150
critic_beta = 0.25
entropy_beta = 0.01
gamma = 0.99
lamb = 0.98

lr_a = 1e-5
lr_c = 2e-5

## Save/Summary
update_frequency = 32
epochs = 20
minibatch_size = 300
save_network_frequency = 200 - (200%update_frequency)
save_stat_frequency = 10 * update_frequency
moving_average_step = 10


# Env Settings
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6
in_size = [None, vision_dx, vision_dy, nchannel]
nenv = 8

# TF SESSION
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, inter_op_parallelism_threads=nenv)

sess = tf.Session(config=config)
progbar = tf.keras.utils.Progbar(total_episodes,interval=10)

# ENVIRONMENT
env = gym.make("cap-v0").unwrapped
env.reset(
    map_size=map_size,
    policy_red=policy.roomba.PolicyGen(env.get_map, env.get_team_red),
    config_path='setting1.ini',
)

# POLICY
targ_AC = AC(
    in_size=in_size,
    action_size=action_space,
    lr_actor=lr_a,
    lr_critic=lr_c,
    scope='global',
    entropy_beta = 0.01,
    sess=sess,
)
AC = AC(
    in_size=in_size,
    action_size=action_space,
    lr_actor=lr_a,
    lr_critic=lr_c,
    scope='main',
    entropy_beta = 0.01,
    global_network = targ_AC,
    sess=sess,
    tau=0.8
)

global_episodes = 0

# Local configuration parameters
def rollout(episode=10):
    episode_rewards = []
    win_rate = []

    step_counter = 0
    traj_list = []
    for _ in range(episode):
        # Initialization
        s1 = env.reset()
        s1 = one_hot_encoder(env.get_obs_blue, env.get_team_blue, vision_range)
        
        # parameters 
        episode_reward = 0 # Episodic Reward
        prev_reward = 0

        # Experience buffer
        trajs = [Trajectory(depth=6) for _ in range(num_blue)]
        
        # Bootstrap
        a1, v1 = AC.run_network(s1)
        was_alive = [ag.isAlive for ag in env.get_team_blue]
        for step in range(step_limit+1):
            s0 = s1
            action, v0 = a1, v1
            
            s1, rc, done, info = env.step(action)
            s1 = one_hot_encoder(env.get_obs_blue, env.get_team_blue, vision_range)
            is_alive = [ag.isAlive for ag in env.get_team_blue]
            reward = (rc - prev_reward - 0.5)

            if step == step_limit and done == False:
                reward = -100
                rc = -100
                done = True
            reward /= 100.0
            episode_reward += reward

            a1, v1 = AC.run_network(s0)

            # push to buffer
            for idx, agent in enumerate(env.get_team_blue):
                if was_alive[idx]:
                    trajs[idx].append([s0[idx], action[idx], reward, v0[idx]])

            # Iteration
            prev_reward = rc
            was_alive = is_alive

            step_counter += 1

            if done:
                break

        traj_list.extend(trajs)
        episode_rewards.append(episode_reward)
        win_rate.append(env.blue_win)

        gae(traj_list, normalize=False)

    return traj_list, step_counter, np.mean(episode_rewards), np.mean(win_rate)

def gae(traj_list, normalize=True):
    for idx, traj in enumerate(traj_list):
        if len(traj) == 0:
            continue
        rewards = np.array(traj[2])
        values = np.array(traj[3])
        
        value_ext = np.append(values, [0])
        td_target  = rewards + gamma * value_ext[1:]
        advantages = rewards + gamma * value_ext[1:] - value_ext[:-1]
        advantages = discount_rewards(advantages, gamma*lamb)

        if normalize:
            advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-6)

        traj_list[idx][4] = td_target.tolist()
        traj_list[idx][5] = advantages.tolist()

def work(saver, writer):
    global global_episodes
    ma_episode_rewards = MovingAve(moving_average_step)
    ma_length = MovingAve(moving_average_step)
    ma_succeed = MovingAve(moving_average_step)

    total_step = 1
            
    policy_red_roomba = policy.roomba.PolicyGen(env.get_map, env.get_team_red)

    # loop
    print('work initiated')
    while global_episodes < total_episodes:
        flag_log = global_episodes % save_stat_frequency == 0 and global_episodes != 0

        traj_list, step_counter, episode_reward, win_rate = rollout(episode=update_frequency)

        aloss, closs, entropy, kl, weight_summary = train(traj_list, epoch=epochs, log=flag_log)
                
        ma_episode_rewards.append(episode_reward)
        ma_length.append(step_counter / update_frequency)
        ma_succeed.append(win_rate)

        # Iterate
        sess.run(global_step_next)
        total_step += step_counter
        global_episodes += update_frequency
        progbar.update(global_episodes)

        if flag_log:
            record({
                'Records/mean_length': ma_length(),
                'Records/mean_succeed': ma_succeed(),
                'Records/mean_episode_reward': ma_episode_rewards(),
                'summary/actor_loss': aloss,
                'summary/critic_loss': closs,
                'summary/Entropy': entropy,
                'summary/KL': kl
            }, writer, weight_summary)
            
        if global_episodes % save_network_frequency == 0 and global_episodes != 0:
            saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

def record(item, writer, weight_summary):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary, global_episodes)
    writer.add_summary(weight_summary, global_episodes)
    writer.flush()

def train(trajs, epoch=10, log=False):
    global writer
    buffer_s, buffer_a, buffer_tdtarget, buffer_adv = [], [], [], []
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        buffer_s.extend(traj[0])
        buffer_a.extend(traj[1])
        buffer_tdtarget.extend(traj[4])
        buffer_adv.extend(traj[5])
    if len(buffer_s) == 0:
        return

    # Update Buffer (minibatch)
    for _ in range(epoch):
        batch_size = min(minibatch_size, len(buffer_s))
        indices = np.random.choice(len(buffer_s), batch_size, replace=False)
        aloss, closs, etrpy, kl = AC.update_global(
            np.array(buffer_s)[indices],
            np.array(buffer_a)[indices],
            np.array(buffer_tdtarget)[indices],
            np.array(buffer_adv)[indices],
            return_kl = True,
	    writer=writer,
	    log=log
        )

    feed_dict = {
        AC.state_input : np.array(buffer_s)[indices],
        AC.action_ : np.array(buffer_a)[indices],
        AC.td_target_ : np.array(buffer_tdtarget)[indices],
        AC.advantage_ : np.array(buffer_adv)[indices]
    }

    weight_summary = AC.sess.run(merged_summary_op, feed_dict)

    AC.pull_global()

    return aloss, closs, etrpy, kl, weight_summary

# ## Run
# Global Network
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, update_frequency)

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3, var_list=AC.get_vars+[global_step])
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    
ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Load Model : ", ckpt.model_checkpoint_path)
    iuv(sess)
else:
    sess.run(tf.global_variables_initializer())
    print("Initialized Variables")

global_episodes = sess.run(global_step) # Reset the counter
saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes) # Initial save

# Summarize
histo_summary = []
for var in AC.get_vars:
    histo_summary.append(tf.summary.histogram(var.name, var))
merged_summary_op = tf.summary.merge(histo_summary)

with sess.as_default(), sess.graph.as_default():
    work(saver, writer)
