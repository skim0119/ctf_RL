
import os
import configparser
import shutil

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
import policy.policy_A3C
import policy.zeros

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory

from network.a3c import ActorCritic as AC
from network.base import initialize_uninitialized_vars as iuv

OVERRIDE = False;
TRAIN_NAME='ATT_9'
LOG_PATH='./logs/'+TRAIN_NAME
MODEL_PATH='./model/' + TRAIN_NAME
GPU_CAPACITY=0.5 # gpu capacity in percentage

if OVERRIDE:
    #  Remove and reset log and model directory
    #  !rm -rf logs/A3C_benchmark/ model/A3C_benchmark
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

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



# ## Hyperparameters

# In[4]:


# Importing global configuration
config = configparser.ConfigParser()
config.read('config.ini')

## Environment
action_space = config.getint('DEFAULT','ACTION_SPACE')
n_agent = 2 #config.getint('DEFAULT','NUM_AGENT')
map_size = config.getint('DEFAULT','MAP_SIZE')
vision_range = 9 #config.getint('DEFAULT','VISION_RANGE')

## Training
total_episodes = 2e5 #config.getint('TRAINING','TOTAL_EPISODES')
max_ep = config.getint('TRAINING','MAX_STEP')
critic_beta = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta = config.getfloat('TRAINING', 'ENTROPY_BETA')
gamma = config.getfloat('TRAINING', 'DISCOUNT_RATE')

decay_lr = config.getboolean('TRAINING','DECAYING_LR')
lr_a = 5e-5#config.getfloat('TRAINING','LR_ACTOR')
lr_c = 2e-4#config.getfloat('TRAINING','LR_CRITIC')

## Save/Summary
save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ')
moving_average_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')

## GPU
gpu_capacity = config.getfloat('GPU_CONFIG','GPU_CAPACITY')
gpu_allowgrow = config.getboolean('GPU_CONFIG', 'GPU_ALLOWGROW')

# In[ ]:


# Local configuration parameters
update_frequency = 64
po_transition = 5000000 # Partial observable

# Env Settings
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6
in_size = [None,vision_dx,vision_dy,nchannel]
nenv = 8#(int) (multiprocessing.cpu_count())

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

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
progbar = tf.keras.utils.Progbar(total_episodes,interval=1)

# ## Worker


def reward_shape(info, done):
    # Reward Expansion
    strategy_reward = np.zeros(3)
    
    # Attack (C/max enemy)
    if len(info['red_alive']) <= 1:
        prev_num_enemy = sum(info['red_alive'][-1])
    else:        
        prev_num_enemy = sum(info['red_alive'][-2])
    num_enemy = sum(info['red_alive'][-1])
    r = int(prev_num_enemy - num_enemy)# / 4
    return r

class Worker(object):
    def __init__(self, name, globalAC, sess, global_step=0):
        # Initialize Environment worker
        self.env = gym.make("cap-v0").unwrapped
        self.env.num_blue_ugv = n_agent
        self.env.num_red_ugv = 4
        self.env.sparse_reward = True
        self.env.red_partial_visibility = False
        self.env.reset(
            map_size=map_size,
            #policy_red=policy.policy_A3C.PolicyGen(self.env.get_map, self.env.get_team_red, color='red')
            policy_red=policy.roomba.PolicyGen(self.env.get_map, self.env.get_team_red)
        )
        self.env()
        self.name = name
        
        # Create AC Network for Worker
        self.AC = AC(in_size=in_size,
                     action_size=action_space,
                     lr_actor=lr_a,
                     lr_critic=lr_c,
                     scope=self.name,
                     entropy_beta = 0.01,
                     sess=sess,
                     global_network=global_ac)
        
        self.sess=sess
        
    def get_action(self, states):
        actions, values = [], []
        for state in states:
            action, value = self.AC.run_network(state[np.newaxis,:])
            #feed_dict = {self.AC.state_input : state[np.newaxis,:]}
            #action, value = self.AC.run_network(feed_dict)
            actions.append(action[0])
            values.append(value[0])
#        feed_dict = {self.AC.state_input : s}
#        a1, v1, _ = self.AC.run_network(feed_dict)
        
#        return a1, v1, _
        return actions, values

    def work(self, saver, writer):
        global global_rewards, global_ep_rewards, global_episodes, global_length, global_succeed
        total_step = 1

        policy_red_a3c = policy.policy_A3C.PolicyGen(self.env.get_map, self.env.get_team_red, color='red')
        policy_red_roomba = policy.roomba.PolicyGen(self.env.get_map, self.env.get_team_red)
                
        # loop
        with self.sess.as_default(), self.sess.graph.as_default():
            while not coord.should_stop() and global_episodes < total_episodes:
                # select red
                if global_episodes < 2e4:
                    policy_red = policy_red_roomba
                else:
                    policy_red = policy_red_a3c
                s0 = self.env.reset(policy_red=policy_red)

                if po_transition < global_episodes:
                    s0 = one_hot_encoder(s0, self.env.get_team_blue, vision_range)
                else:
                    s0 = one_hot_encoder(self.env._env, self.env.get_team_blue, vision_range)
                
                # parameters 
                ep_r = 0 # Episodic Reward
                prev_r = 0
                was_alive = [ag.isAlive for ag in self.env.get_team_blue]

                trajs = [Trajectory(depth=4) for _ in range(n_agent)]
                
                # Bootstrap
                a1, v1 = self.get_action(s0)
                for step in range(max_ep+1):
                    a, v0 = a1, v1
                    
                    s1, rc, d, info = self.env.step(a)
                    r_shaped = reward_shape(info, d)
                    if po_transition < global_episodes:
                        s1 = one_hot_encoder(s1, self.env.get_team_blue, vision_range)
                    else:
                        s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, vision_range)
                    is_alive = [ag.isAlive for ag in self.env.get_team_blue]
                    r = (rc - prev_r-0.5)

                    if step == max_ep and d == False:
                        r = -100
                        rc = -100
                        d = True

                    r /= 100.0
                    r = r_shaped
                    rc = r_shaped
                    ep_r += r

                    if d:
                        v1 = [0.0 for _ in range(len(self.env.get_team_blue))]
                    else:
                        a1, v1 = self.get_action(s1)

                    # push to buffer
                    for idx, agent in enumerate(self.env.get_team_blue):
                        if was_alive[idx]:
                            trajs[idx].append([s0[idx],
                                               a[idx],
                                               r_shaped,
                                               v0[idx]
                                              ])

                    if total_step % update_frequency == 0 or d:
                        aloss, closs, etrpy, feed_dict = self.train(trajs, sess, v1)
                        trajs = [Trajectory(depth=5) for _ in range(n_agent)]

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
                    summary_ = sess.run(merged_summary_op, feed_dict)
                    summary = tf.Summary()
                    summary.value.add(tag='Records/mean_reward', simple_value=global_rewards())
                    summary.value.add(tag='Records/mean_length', simple_value=global_length())
                    summary.value.add(tag='Records/mean_succeed', simple_value=global_succeed())
                    summary.value.add(tag='Records/mean_episode_reward', simple_value=global_ep_rewards())
                    summary.value.add(tag='summary/Entropy', simple_value=etrpy)
                    summary.value.add(tag='summary/actor_loss', simple_value=aloss)
                    summary.value.add(tag='summary/critic_loss', simple_value=closs)
                    writer.add_summary(summary,global_episodes)
                    writer.add_summary(summary_,global_episodes)

                    writer.flush()
                if global_episodes % save_network_frequency == 0 and global_episodes != 0:
                    saver.save(self.sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

    def train(self, trajs, sess, bootstrap=0.0):
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
            

            #feed_dict = {
            #    self.AC.state_input : observations,
            #    self.AC.action_ : actions,
            #    self.AC.td_target_ : td_target,
            #    self.AC.advantage_ : advantages,
            #}

            # Update Buffer
            #aloss, closs, etrpy = self.AC.update_global(feed_dict)

            # get global parameters to local ActorCritic 
            #self.AC.pull_global()
            
            buffer_s.extend(observations)
            buffer_a.extend(actions)
            buffer_tdtarget.extend(td_target.tolist())
            buffer_adv.extend(advantages.tolist())

        buffer_s, buffer_a, buffer_tdtarget, buffer_adv = np.stack(buffer_s), np.array(buffer_a), np.array(buffer_tdtarget), np.array(buffer_adv)
        feed_dict = {
            self.AC.state_input : buffer_s,
            self.AC.action_ : buffer_a,
            self.AC.td_target_ : buffer_tdtarget,
            self.AC.advantage_ : buffer_adv,
        }

        # Update Buffer
        aloss, closs, etrpy = self.AC.update_global(
            buffer_s, buffer_a, buffer_tdtarget, buffer_adv)
            #feed_dict)

        # get global parameters to local ActorCritic 
        self.AC.pull_global()
        
        return aloss, closs, etrpy, feed_dict
    

# ## Run

# In[ ]:


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
    
coord = tf.train.Coordinator()
worker_threads = []
global_episodes = sess.run(global_step)

saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

# Summarize
for var in tf.trainable_variables(scope=global_scope):
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()

for worker in workers:
    job = lambda: worker.work(saver, writer)
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)
