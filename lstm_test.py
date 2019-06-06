"""
Extendedd training:
    1. include UAV
    2. train on partial observable space.
    3. assume infinite bandwidth between all agent
        - This assumption is subjected to change as the difficulty level of the envrionment is adjusted
        - For now, assume the easiest case possible.
    4. different policy for ground and air vehicle

TODO:
    1. Until the environment allows separate policy for UAV and UGV, self-play will not be used.
"""

import os
import stat
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
import policy.ground_a3c

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory

from network.a3c import ActorCritic as AC

from network.base import initialize_uninitialized_vars as iuv

OVERRIDE = True;
TRAIN_NAME='lstm_predict_test_2'
LOG_PATH='./logs/'+TRAIN_NAME
GPU_CAPACITY=0.5 # gpu capacity in percentage

if OVERRIDE:
    #  Remove and reset log and model directory
    #  !rm -rf logs/A3C_benchmark/ model/A3C_benchmark
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH,ignore_errors=True)

# Create model and log directory
if not os.path.exists(LOG_PATH):
    try:
        os.makedirs(LOG_PATH)
    except OSError:
        raise OSError(f'Creation of the directory {LOG_PATH} failed')

## Hyperparameters
# Importing global configuration
config = configparser.ConfigParser()
config.read('config.ini')

## Environment
action_space = 5 
num_blue = 4
map_size = 20
vision_range = 19 

total_episodes = int(2e5)
max_ep = 150

lr = 1e-4

## Save/Summary
save_stat_frequency = 128

# Env Settings
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6
in_size = [None, vision_dx, vision_dy, nchannel]
nenv = 8

# Asynch Settings
global_scope = 'global'

# Launch the session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, inter_op_parallelism_threads=nenv)

lstm_graph = tf.Graph()
sess = tf.Session(config=config, graph=lstm_graph)
with lstm_graph.as_default():
    encoded_state = tf.placeholder(tf.float32, [None, 1024])

    #Recurrent network for temporal dependencies
    lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=1024, state_is_tuple=True)
    c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
    h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
    c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
    h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
    rnn_in = tf.expand_dims(encoded_state, [0])
    step_size = tf.shape(encoded_state)[:1]
    state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm_cell,
        rnn_in,
        initial_state=state_in,
        sequence_length=step_size,
        time_major=False)
    lstm_c, lstm_h = lstm_state
    state_out_tuple = (lstm_c[:1, :], lstm_h[:1, :])
    rnn_out = tf.reshape(lstm_outputs, [-1, 1024])

    encoded_predict = tf.placeholder(tf.float32, [None,1024])
    loss = tf.losses.mean_squared_error(rnn_out, encoded_predict)
    train = tf.train.AdamOptimizer(lr).minimize(loss)

    sess.run(tf.global_variables_initializer())

    print("Initialized Variables")

progbar = tf.keras.utils.Progbar(total_episodes,interval=1)

# ## Worker
env = gym.make("cap-v0").unwrapped
env.red_partial_visibility = True
env.reset(
    map_size=map_size,
    policy_red=policy.roomba.PolicyGen(env.get_map, env.get_team_red)
)

# ## Run
# Global Network
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, 1)

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3)
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

def record(item, step):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary, step)
    writer.flush()
    

policy_blue = policy.ground_a3c.PolicyGen(env.get_map, env.get_team_blue)

for global_episodes in range(total_episodes):
    s0 = env.reset()
    s0 = one_hot_encoder(env.get_obs_blue, env.get_team_blue, vision_range)
    
    # parameters 
    was_alive = [ag.isAlive for ag in env.get_team_blue]
    trajs = [Trajectory(depth=1) for _ in range(num_blue)]
    for idx, agent in enumerate(env.get_team_blue):
        trajs[idx].append([s0[idx]])
    
    # Bootstrap
    action = policy_blue.gen_action(env.get_team_blue, env.get_obs_blue)
    for step in range(max_ep+1):
        state, reward, done, info = env.step(action)
        state = one_hot_encoder(env.get_obs_blue, env.get_team_blue, vision_range)
        is_alive = [ag.isAlive for ag in env.get_team_blue]

        action = policy_blue.gen_action(env.get_team_blue, env.get_obs_blue)

        # push to buffer
        for idx, agent in enumerate(env.get_team_blue):
            if was_alive[idx]:
                trajs[idx].append([state[idx]])

        # Iteration
        was_alive = is_alive

        if done:
            break
            
    with lstm_graph.as_default():
        losses = []
        for traj in trajs:
            if len(traj) == 0:
                continue
            observations = traj[0]
            encoded_obs = policy_blue.query_encoded_state(observations)
            feed_dict = {
                    encoded_state: encoded_obs[:-1],
                    c_in: c_init,
                    h_in: h_init,
                    encoded_predict: encoded_obs[1:]
                    }
            mse_loss, _ = sess.run([loss, train], feed_dict=feed_dict)
            losses.append(mse_loss)
    if global_episodes % save_stat_frequency == 0 and global_episodes != 0:
        record({'Predict_Loss/MSE' : np.mean(losses)}, global_episodes)

    progbar.update(global_episodes)

