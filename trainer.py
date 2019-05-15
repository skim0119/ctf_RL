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

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory

from network.a3c import ActorCritic as AC

from network.base import initialize_uninitialized_vars as iuv

OVERRIDE = True;
TRAIN_NAME='partial'
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
# Importing global configuration
config = configparser.ConfigParser()
config.read('config.ini')

## Environment
action_space = 5 
num_blue = 4
num_uav = 2
map_size = 20
vision_range = 19 

## Training
total_episodes = 2e5
max_ep = 150
gamma = 0.98

lr_a = 5e-5
lr_c = 2e-4

## Save/Summary
save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ')
moving_average_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')


# Local configuration parameters
update_frequency = 32

# Env Settings
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6
in_size = [None, vision_dx, vision_dy, nchannel]
nenv = 8

# Asynch Settings
global_scope = 'global'

# ## Environment Setting

global_rewards = MA(moving_average_step)
global_ep_rewards = MA(moving_average_step)
global_length = MA(moving_average_step)
global_succeed = MA(moving_average_step)
global_episodes = 0

# Launch the session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options, inter_op_parallelism_threads=nenv)

sess = tf.Session(config=config)
progbar = tf.keras.utils.Progbar(total_episodes,interval=1)

# ## Worker
class Worker(object):
    def __init__(self, name, global_ac, global_av, sess, global_step=0):
        # Initialize Environment worker
        self.env = gym.make("cap-v0").unwrapped
        self.env.red_partial_visibility = True
        self.env.reset(
            map_size=map_size,
            policy_red=policy.roomba.PolicyGen(self.env.get_map, self.env.get_team_red)
        )
        self.env.reset()
        self.name = name
        
        # Create AC Network for Worker
        self.AC_av = AC(
            in_size=in_size,
            action_size=action_space,
            lr_actor=lr_a,
            lr_critic=lr_c,
            scope=self.name+'_av',
            entropy_beta = 0.01,
            sess=sess,
            global_network=global_av
        )

        self.AC = AC(
            in_size=in_size,
            action_size=action_space,
            lr_actor=lr_a,
            lr_critic=lr_c,
            scope=self.name,
            entropy_beta = 0.01,
            sess=sess,
            global_network=global_ac
        )

        self.sess=sess

    def get_action(self, states):
        uav_states, ugv_states = states[:num_uav,:], states[num_uav:,:]
        actions_ugv, values_ugv = self.AC.run_network(ugv_states)
        actions_uav, values_uav = self.AC_av.run_network(uav_states)
        return np.concatenate([actions_uav, actions_ugv],axis=None), np.concatenate([values_uav, values_ugv],axis=None)

    def work(self, saver, writer, coord):
        global global_rewards, global_ep_rewards, global_episodes, global_length, global_succeed
        total_step = 1
                
        policy_red_roomba = policy.roomba.PolicyGen(self.env.get_map, self.env.get_team_red)
        policy_red_zeros = policy.zeros.PolicyGen(self.env.get_map, self.env.get_team_red)

        # loop
        print(f'{self.name} work initiated')
        with self.sess.as_default(), self.sess.graph.as_default(), coord.stop_on_exception():
            while not coord.should_stop() and global_episodes < total_episodes:
                # select red
                if global_episodes < 2e4:
                    policy_red = policy_red_zeros
                elif global_episodes < 5e4:
                    policy_red = policy_red_roomba
                '''else:
                    policy_red = policy_red_a3c
                    if total_step % 500*150:
                        policy_red_a3c.reset_network_weight()
                        '''

                s0 = self.env.reset(policy_red=policy_red)
                s0 = one_hot_encoder(s0, self.env.get_team_blue, vision_range)
                
                # parameters 
                ep_r = 0 # Episodic Reward
                prev_r = 0
                was_alive = [ag.isAlive for ag in self.env.get_team_blue]

                trajs_av = [Trajectory(depth=4) for _ in range(num_uav)]
                trajs_gv = [Trajectory(depth=4) for _ in range(num_blue)]
                
                # Bootstrap
                a1, v1 = self.get_action(s0)
                for step in range(max_ep+1):
                    a, v0 = a1, v1
                    
                    s1, rc, d, info = self.env.step(a.tolist())
                    s1 = one_hot_encoder(s1, self.env.get_team_blue, vision_range)
                    is_alive = [ag.isAlive for ag in self.env.get_team_blue]
                    r = (rc - prev_r - 0.5)

                    if step == max_ep and d == False:
                        r = -100
                        rc = -100
                        d = True

                    r /= 100.0
                    ep_r += r

                    if d:
                        v1 = [0.0 for _ in range(num_blue+num_uav)]
                    else:
                        a1, v1 = self.get_action(s1)

                    # push to buffer
                    for idx, agent in enumerate(self.env.get_team_blue):
                        if was_alive[idx]:
                            if idx < num_uav:
                                trajs_av[idx].append([s0[idx], a[idx], r, v0[idx]])
                            else:
                                trajs_gv[idx-num_uav].append([s0[idx], a[idx], r, v0[idx]])

                    if total_step % update_frequency == 0 or d:
                        self.train(trajs_av, v1[:num_uav], self.AC_av)
                        self.train(trajs_gv, v1[num_uav:], self.AC)
                        trajs_av = [Trajectory(depth=4) for _ in range(num_uav)]
                        trajs_gv = [Trajectory(depth=4) for _ in range(num_blue)]

                    # Iteration
                    prev_r = rc
                    was_alive = is_alive
                    s0=s1
                    total_step += 1

                    if d:
                        break
                        
                global_ep_rewards.append(ep_r)
                global_rewards.append(rc)
                global_length.append(step)
                global_succeed.append(self.env.blue_win)
                global_episodes += 1
                self.sess.run(global_step_next)
                progbar.update(global_episodes)

                if global_episodes % save_stat_frequency == 0 and global_episodes != 0:
                    self.record({
                        'Records/mean_reward': global_rewards(),
                        'Records/mean_length': global_length(),
                        'Records/mean_succeed': global_succeed(),
                        'Records/mean_episode_reward': global_ep_rewards(),
                    }, writer)
                    
                if global_episodes % save_network_frequency == 0 and global_episodes != 0:
                    saver.save(self.sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

    def record(self, item, writer):
        summary = tf.Summary()
        for key, value in item.items():
            summary.value.add(tag=key, simple_value=value)
        writer.add_summary(summary,global_episodes)
        writer.flush()

    def train(self, trajs, bootstrap=0.0, network=None):
        buffer_s, buffer_a, buffer_tdtarget, buffer_adv = [], [], [], []
        for idx, traj in enumerate(trajs):
            if len(traj) == 0:
                continue
            observations = traj[0]
            actions = traj[1]
            rewards = np.array(traj[2])
            values = np.array(traj[3])
            
            value_ext = np.append(values, [bootstrap[idx]])
            td_target  = rewards + gamma * value_ext[1:]
            advantages = rewards + gamma * value_ext[1:] - value_ext[:-1]
            advantages = discount_rewards(advantages,gamma)
            
            buffer_s.extend(observations)
            buffer_a.extend(actions)
            buffer_tdtarget.extend(td_target.tolist())
            buffer_adv.extend(advantages.tolist())

        if len(buffer_s) == 0:
            return

        feed_dict = {
            network.state_input : np.stack(buffer_s),
            network.action_ : np.array(buffer_a),
            network.td_target_ : np.array(buffer_tdtarget),
            network.advantage_ : np.array(buffer_adv),
        }

        # Update Buffer
        aloss, closs, etrpy = network.update_global(
            buffer_s,
            buffer_a,
            buffer_tdtarget,
            buffer_adv
        )

        # get global parameters to local ActorCritic 
        network.pull_global()

        return# aloss, closs, etrpy

# ## Run
# Global Network
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, 1)
global_ac = AC(in_size=in_size, action_size=action_space, scope=global_scope, sess=sess)
global_av = AC(in_size=in_size, action_size=action_space, scope=global_scope+'_av', sess=sess)

# Local workers
workers = []
# loop for each workers
for idx in range(nenv):
    name = 'W_%i' % idx
    workers.append(Worker(name, global_ac, global_av, sess, global_step=global_step))
    print(f'worker: {name} initiated')

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3, var_list=global_ac.get_vars+[global_step]+global_av.get_vars)
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
for var in tf.trainable_variables(scope=global_scope):
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()

try:
    coord = tf.train.Coordinator()
    threads = []
    for worker in workers:
        thread = threading.Thread(target=worker.work, args=(saver, writer, coord))
        thread.start()
        threads.append(thread)
    time.sleep(2)
except Exception as exception:
    print(f'End call used: {exception}\n')
    coord.request_stop(exception)
finally:
    coord.join(threads)
    print('EOP\n')
    coord.request_stop()
