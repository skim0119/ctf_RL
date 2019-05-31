"""
This module is written to run training. 

TODO:
    - Transfer learning
    - Differentiate training env.

THS:
    - Ask if new train or rewrite previous
        - if rewrite, read directory list (model/log)
        - Prompt if want to continue
        - Save train_name

"""

import os
import shutil
import time

import gym

import numpy as np
import random
import math

# the modules that you can use to generate the policy.
import policy.random
import policy.roomba
import policy.zeros
import policy.policy_A3C
import policy.policy_ensemble

import inquirer
import re
from tqdm import tqdm

import tensorflow as tf

# utilities
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards, store_args
from utility.RL_Wrapper import TrainedNetwork
from utility.RL_Wrapper import Subgraph
from utility.buffer import Trajectory

# Training network
from network.DQN import DQN as Network
from network.a3c import ActorCritic
from network.base import initialize_uninitialized_vars as iuv

answer = inquirer.prompt([
    inquirer.List(
        'new_model',
        message='Create New Policy [(Y)es/(N)o] ',
        choices=['Yes', 'No']
    )
])
NEW_MODEL = (answer['new_model'] == 'Yes')

existings = os.listdir('model')
if NEW_MODEL:
    answer = inquirer.prompt([
        inquirer.Text(
            'model_name',
            message='Model Name  '
        )
    ])
    if answer['model_name'] in existings:
        raise NameError(f'Train name already exist. Please try new name or override existing one')
    OVERRIDE = False
else:
    answer = inquirer.prompt([
        inquirer.List(
            'model_name',
            message='Select pre-existing model ',
            choices=existings
        ),
        inquirer.List(
            'override',
            message='Override pre-existing model?  ',
            choices=['Yes', 'No'],
        )
    ])
    OVERRIDE = answer['override'] == 'Yes'

TRAIN_NAME = answer['model_name']

LOG_PATH = './logs/' + TRAIN_NAME
MODEL_PATH = './model/' + TRAIN_NAME
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

# Environment
env = gym.make("cap-v0").unwrapped  # initialize the environment
#policy_red = policy.roomba.PolicyGen(env.get_map, env.get_team_red)
policy_red = policy.policy_A3C.PolicyGen(env.get_map, env.get_team_red, color='red')

print('---- Environment is Ready ----')

print('---- Train Setting')
'''
questions = [
    inquirer.Text(
        'run_episode',
        message="How many episode to train? (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='100000',
    ),
    inquirer.Text(
        'episode_length',
        message="Length of each episode (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='150',
    ),
    inquirer.List(
        'run_partial_red',
        message="Red play with partial observability?",
        choices=['yes', 'no'],
    ),
    inquirer.List(
        'run_partial_blue',
        message="Blue play with partial observability?",
        choices=['yes', 'no'],
    ),
    inquirer.Text(
        'num_red',
        message="How many RED agents? (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='4',
    ),
    inquirer.Text(
        'num_blue',
        message="How many BLUE agents? (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='4',
    ),
    inquirer.Text(
        'map_size',
        message="Map Size? (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='20',
    ),
]
params = inquirer.prompt(questions)
env.red_partial_visibility = params['run_partial_red'] == 'yes'
env.num_blue_ugv = int(params['num_blue'])
env.num_red_ugv = int(params['num_red'])
env.reset(map_size=int(params['map_size']), policy_red=policy_red)
'''
env.red_partial_visibility = False
env.num_blue_ugv = 4
env.num_red_ugv = 4
env.reset(map_size=20, policy_red=policy_red)
env()


# Run
start_time = time.time()
total_episodes = 200000
max_ep = 150
partial_visible = False
'''
total_episodes = int(params['run_episode'])
max_ep = int(params['episode_length'])
partial_visible = params['run_partial_blue'] == 'yes'
'''

# Replay Variables
update_frequency = 20
batch_size = 2000
replay_capacity = 5000

# Saving Related
save_network_frequency = 1000
save_stat_frequency = 128
moving_average_step = 128

# Training Variables
lr_a = 1e-3  # Learning Rate
gamma = 0.98  # Discount Factor
tau = 0.05

# Env Settings
VISION_RANGE = 9  # What decide the network size !!!
VISION_dX, VISION_dY = 2 * VISION_RANGE + 1, 2 * VISION_RANGE + 1
N_CHANNEL = 6
in_size = [None, VISION_dX, VISION_dY, N_CHANNEL]

action_size = 2
num_blue = len(env.get_team_blue)
num_red = len(env.get_team_red)

global_rewards = MA(moving_average_step)
global_ep_rewards = MA(moving_average_step)
global_length = MA(moving_average_step)
global_succeed = MA(moving_average_step)
global_episodes = 0

# Launch the session
gpu_capacity = 0.3
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_capacity, allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess_nav = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=tf.Graph())
sess_att = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=tf.Graph())

progbar = tf.keras.utils.Progbar(total_episodes, interval=1)

def reward_shape_att(info, done):
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

def reward_shape_nav(info, done):
    # Search
    if info['red_flag_caught'][-1]:
        r = 1
    else:
        r=-1
    '''elif done:
        r = -1
    else:
        r = 0'''
    return r

# Load global nav network
with sess_nav.as_default(), sess_nav.graph.as_default():
    global_nav = ActorCritic(in_size=in_size, action_size=5, scope='global', sess=sess_nav)
    saver = tf.train.Saver(var_list=global_nav.get_vars)
    model_path = './model/NAV' 
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess_nav, ckpt.model_checkpoint_path)
        print("Load Model : ", ckpt.model_checkpoint_path)
        iuv(sess_nav)
    else:
        raise NotImplementedError
        
# Load global att network
with sess_att.as_default(), sess_att.graph.as_default():
    global_att = ActorCritic(in_size=in_size, action_size=5, scope='global', sess=sess_att)
    saver = tf.train.Saver(var_list=global_att.get_vars)
    model_path = './model/ATT_9' 
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess_att, ckpt.model_checkpoint_path)
        print("Load Model : ", ckpt.model_checkpoint_path)
        iuv(sess_att)
    else:
        raise NotImplementedError

#  Worker
class Worker(object):
    @store_args
    def __init__(self, name, target_network, sess, trainer, env, global_step=0):
        # Create AC Network for Worker
        self.Network = Network(
            in_size=in_size,
            action_size=action_size,
            scope=name,
            trainer=trainer,
            num_agent=num_blue,
            tau=tau,
            gamma=gamma,
            grad_clip_norm=0,
            global_step=global_step,
            sess=sess,
            target_network=target_network,
        )
        self.sub_policy = [
            Subgraph(in_size, model_name=name, network_type=ActorCritic, sess=sess_att, global_network=global_att),
            Subgraph(in_size, model_name=name, network_type=ActorCritic, sess=sess_nav, global_network=global_nav)
            # TrainedNetwork(model_name='ATT_9'),
            # TrainedNetwork(model_name='NAV')
        ]

    def get_action(self, raw_observation, agent_list, process_ids):
        states = one_hot_encoder(raw_observation, agent_list, VISION_RANGE)
        choices = [self.sub_policy[0].get_action(states), self.sub_policy[1].get_action(states)]

        # Arbitrary
        action_out = [choices[pid][0][aid] for aid, pid in enumerate(process_ids)]
        value_out = [choices[pid][1][aid] for aid, pid in enumerate(process_ids)]

        return action_out, value_out

    def work(self, saver, writer):
        global global_rewards, global_ep_rewards, global_episodes, global_length, global_succeed
        total_step = 1
        local_ep = 0
        buffer = Experience_buffer(experience_shape=6,
                                   buffer_size=replay_capacity)
        epsilon = 1.0
        epsilon_gamma = 0.999
        epsilon_final = 0.1
        while global_episodes < total_episodes:
            local_ep += 1
            raw_obs = self.env.reset()
            if partial_visible:
                s1 = one_hot_encoder(raw_obs, self.env.get_team_blue, VISION_RANGE)
            else:
                s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, VISION_RANGE)

            # parameters
            ep_r = 0
            prev_r = 0
            is_alive = [True] * num_blue

            episode_buffer = []

            trajs_att = [Trajectory(depth=4) for _ in range(num_blue)]
            trajs_nav = [Trajectory(depth=4) for _ in range(num_blue)]

            for step in range(max_ep + 1):
                # Set sub-policy
                if step % 15 == 0:
                    with self.sess.as_default(), self.sess.graph.as_default():
                        pids = self.Network.run_network(np.expand_dims(s1, axis=0))[0]

                v0 = [0 for _ in range(num_blue)]
                if random.random() < epsilon:
                    # Random Exploration
                    a = random.choices(range(action_size), k=4)
                    epsilon = max(epsilon_final, epsilon * epsilon_gamma)
                else:
                    a,v0 = self.get_action(raw_obs, self.env.get_team_blue, pids)

                s0 = s1
                raw_obs, rc, d, info = self.env.step(a)
                r_nav = reward_shape_nav(info, d)
                r_att = reward_shape_att(info, d)
                if partial_visible:
                    s1 = one_hot_encoder(raw_obs, self.env.get_team_blue, VISION_RANGE)
                else:
                    s1 = one_hot_encoder(self.env._env, self.env.get_team_blue, VISION_RANGE)
                is_alive = info['blue_alive'][-1]

                r = (rc - prev_r - 0.01)
                if step == max_ep and not d:
                    r = -100
                    rc = -100
                    d = True

                r /= 100.0
                ep_r += r

                # push to buffer
                for idx in range(num_blue):
                    if step > 0:
                        was_alive = info['blue_alive'][-2]
                    else:
                        was_alive = [True] * num_blue
                    if was_alive[idx]:
                        episode_buffer.append([s0, a, r, s1, d, is_alive * 1])

                # push to buffer for sub-policies
                if d:
                    v1 = [0.0 for _ in range(len(self.env.get_team_blue))]
                else:
                    a1, v1 = self.get_action(raw_obs, self.env.get_team_blue, pids)

                for idx, agent in enumerate(self.env.get_team_blue):
                    if was_alive[idx]:
                        if pids[idx] == 0:
                            trajs_att[idx].append([
                                s0[idx],
                                a[idx],
                                r_att,
                                v0[idx]
                            ])
                        else:
                            trajs_nav[idx].append([
                                s0[idx],
                                a[idx],
                                r_nav,
                                v0[idx]
                            ])

                if total_step % update_frequency == 0 or d:
                    self.sub_policy[0].train(trajs_att, v1)
                    self.sub_policy[1].train(trajs_nav, v1)

                trajs_att = [Trajectory(depth=4) for _ in range(num_blue)]
                trajs_nav = [Trajectory(depth=4) for _ in range(num_blue)]

                # Iteration
                prev_r = rc
                total_step += 1

                if d:
                    buffer.add(episode_buffer)
                    if local_ep % update_frequency == 0 and local_ep > 0:
                        batch = buffer.pop(size=batch_size, shuffle=True)
                        aloss, entropy = self.train(batch)
                        # buffer.flush()
                    break

            global_ep_rewards.append(ep_r)
            global_rewards.append(rc)
            global_length.append(step)
            global_succeed.append(self.env.blue_win)
            global_episodes += 1
            self.sess.run(global_step_next)
            progbar.update(global_episodes)
            if global_episodes % save_stat_frequency == 0 and global_episodes != 0:
                summary = tf.Summary()
                summary.value.add(tag='Records/mean_reward', simple_value=global_rewards())
                summary.value.add(tag='Records/mean_length', simple_value=global_length())
                summary.value.add(tag='Records/mean_succeed', simple_value=global_succeed())
                summary.value.add(tag='Records/mean_episode_reward', simple_value=global_ep_rewards())
                summary.value.add(tag='summary/loss', simple_value=aloss)
                summary.value.add(tag='summary/entropy', simple_value=entropy)
                writer.add_summary(summary, global_episodes)
                writer.flush()
            if global_episodes % save_network_frequency == 0:
                saver.save(self.sess, MODEL_PATH + '/ctf_policy.ckpt', global_step=global_episodes)

    def train(self, batch):
        batch = [*zip(*batch)]
        states0 = np.array(batch[0][:])
        actions = np.array(batch[1][:])
        rewards = discount_rewards(batch[2][:], gamma)
        states1 = np.array(batch[3][:])
        dones = np.array(batch[4][:])
        masks = np.array(batch[5][:])
        with self.sess.as_default(), self.sess.graph.as_default():
            loss, entropy = self.Network.update_full(states0, actions, rewards, states1, dones, masks)

        return loss, entropy


with tf.name_scope('Global_Step'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_next = tf.assign_add(global_step, 1)
trainer = tf.train.AdamOptimizer(learning_rate=lr_a)
target_network = Network(in_size=in_size,
                         action_size=action_size,
                         scope='target',
                         num_agent=num_blue,
                         global_step=global_step)

name = 'primary'
worker = Worker(name=name, sess=sess, trainer=trainer, target_network=target_network, env=env)
print(f'{name} initiated')

saver = tf.train.Saver(max_to_keep=3)
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Load Model : ", ckpt.model_checkpoint_path)
    iuv(sess)
    iuv(sess_nav)
    iuv(sess_att)
else:
    sess.run(tf.global_variables_initializer())
    print("Initialized Variables")

saver.save(sess, MODEL_PATH + '/ctf_policy.ckpt', global_step=global_episodes)
global_episodes = sess.run(global_step)
worker.work(saver, writer)
