import os
import shutil
import stat
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
import policy.roombaV2
import policy.policy_A3C
import policy.zeros

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory
from utility.gae import gae

from method.attention import A3C_attention as AC

from method.base import initialize_uninitialized_vars as iuv

OVERRIDE = False;
TRAIN_NAME = 'golub_defense_a3c'
LOG_PATH='./logs/'+TRAIN_NAME
MODEL_PATH='./model/' + TRAIN_NAME
GPU_CAPACITY=0.9 # gpu capacity in percentage

if OVERRIDE:
    #  Remove and reset log and model directory
    #  !rm -rf logs/A3C_benchmark/ model/A3C_benchmark
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH,ignore_errors=True)
        if os.path.exists(LOG_PATH):
            print('not deleted')

    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH,ignore_errors=True)
        if os.path.exists(MODEL_PATH):
            print('not deleted')

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
map_size = 20
vision_range = 19 

## Training
total_episodes = 1e6
max_ep = config.getint('TRAINING','MAX_STEP')
critic_beta = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta = config.getfloat('TRAINING', 'ENTROPY_BETA')
gamma = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd = 0.98 # GAE constant

lr_a = 5e-5
lr_c = 2e-4

## Save/Summary
save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ') * 4
moving_average_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')


# Local configuration parameters
update_frequency = 32

# Env Settings
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6
in_size = [None, vision_dx, vision_dy, nchannel]
nenv = 16

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
progbar = tf.keras.utils.Progbar(total_episodes,interval=5)

# ## Worker
class Worker(object):
    def __init__(self, name, globalAC, sess, global_step=0):
        # Initialize Environment worker
        self.env = gym.make("cap-v0").unwrapped
        self.env.reset(
            map_size=map_size,
            policy_red=policy.policy_A3C,
            config_path='config_defense.ini'
        )
        self.name = name
        
        # Create AC Network for Worker
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

    def reward_shape(self, done):
        if self.env.blue_flag:
            return -1
        else:
            return 0

    def get_action(self, states):
        actions, values = self.AC.run_network(states)
        return actions, values

    def work(self, saver, writer, coord):
        global global_rewards, global_ep_rewards, global_episodes, global_length, global_succeed
        total_step = 1

        # loop
        print(f'{self.name} work initiated')
        with self.sess.as_default(), self.sess.graph.as_default(), coord.stop_on_exception():
            while not coord.should_stop() and global_episodes < total_episodes:
                log_on = False # global_episodes % save_stat_frequency == 0 and global_episodes > 0

                s0 = self.env.reset()
                s0 = one_hot_encoder(self.env._env, self.env.get_team_blue, vision_range)
                
                # parameters 
                ep_r = 0 # Episodic Reward
                was_alive = [ag.isAlive for ag in self.env.get_team_blue]

                trajs = [Trajectory(depth=4) for _ in range(self.env.NUM_BLUE)]

                #self.AC.reset_rnn(num_memory=self.env.NUM_BLUE)
                
                # Bootstrap
                a1, v1 = self.get_action(s0)
                for step in range(max_ep+1):
                    a, v0 = a1, v1
                    
                    s1, _, d, info = self.env.step(a)
                    s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, vision_range)
                    is_alive = [ag.isAlive for ag in self.env.get_team_blue]
                    r = self.reward_shape(done) - 0.01
                    ep_r += r

                    if step == max_ep:
                        d = True

                    if d:
                        v1 = [0.0 for _ in range(self.env.NUM_BLUE)]
                    else:
                        a1, v1 = self.get_action(s1)

                    # push to buffer
                    for idx, agent in enumerate(self.env.get_team_blue):
                        if was_alive[idx]:
                            trajs[idx].append([s0[idx],
                                               a[idx],
                                               r,
                                               v0[idx]
                                              ])

                    if total_step % update_frequency == 0 or d:
                        self.train(trajs, v1, writer, log_on)
                        trajs = [Trajectory(depth=4) for _ in range(self.env.NUM_BLUE)]

                    # Iteration
                    was_alive = is_alive
                    s0 = s1
                    total_step += 1

                    if d:
                        break
                        
                global_ep_rewards.append(ep_r)
                global_length.append(step)
                global_succeed.append(self.env.blue_win)
                global_episodes += 1
                self.sess.run(global_step_next)
                progbar.update(global_episodes)

                if log_on:
                    self.record({
                        'Records/mean_reward': global_rewards(),
                        'Records/mean_length': global_length(),
                        'Records/mean_succeed': global_succeed(),
                        'Records/mean_episode_reward': global_ep_rewards(),
                    }, writer, ignore_nan=True)

                if global_episodes % save_network_frequency == 0 and global_episodes != 0:
                    saver.save(self.sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

    def record(self, item, writer, ignore_nan=False):
        summary = tf.Summary()
        for key, value in item.items():
            if ignore_nan and math.isnan(value):
                continue
            summary.value.add(tag=key, simple_value=value)
        writer.add_summary(summary,global_episodes)
        writer.flush()

    def train(self, trajs, bootstrap=0.0, writer=None, log=False):
        global global_episodes
        train_counter = 0
        for idx, traj in enumerate(trajs):
            if len(traj) <= 1:
                continue
            train_counter += 1

            observations = np.stack(traj[0])
            actions = np.array(traj[1])
            
            # GAE
            td_target, advantages = gae(traj[2], traj[3], bootstrap[idx], gamma, lambd, False)
            
            # Update Buffer
            self.AC.update_global(
                    observations,
                    actions,
                    td_target,
                    advantages,
                    global_episodes,
                    writer,
                    log=log
                )

        # get global parameters to local ActorCritic 
        self.AC.pull_global()

# ## Run
# Global Network
global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, 1)
global_ac = AC(in_size=in_size, action_size=action_space, scope=global_scope, sess=sess)

# Local workers
workers = []
# loop for each workers
for idx in range(nenv):
    name = 'W_%i' % idx
    workers.append(Worker(name, global_ac, sess, global_step=global_step))
    print(f'worker: {name} initiated')

# Resotre / Initialize
saver = tf.train.Saver(max_to_keep=3, var_list=global_ac.get_vars+[global_step])
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
