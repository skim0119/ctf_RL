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
import policy.policy_A3C
import policy.policy_scheduler
import policy.policy_clone
import policy.roombaV2

# Data Processing Module
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.buffer import Trajectory
from utility.gae import gae
from utility.multiprocessing import SubprocVecEnv

from method.ppo import PPO as Network

from method.base import initialize_uninitialized_vars as iuv

## Training Setting

OVERRIDE = False;
TRAIN_NAME='meta'
LOG_PATH='./logs/'+TRAIN_NAME
MODEL_PATH='./model/' + TRAIN_NAME
GPU_CAPACITY=0.90

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

##### Hyperparameters
# Importing global configuration
config = configparser.ConfigParser()
config.read('config.ini')

## Training
total_episodes = 1e6
max_ep = config.getint('TRAINING','MAX_STEP')
critic_beta = config.getfloat('TRAINING', 'CRITIC_BETA')
entropy_beta = config.getfloat('TRAINING', 'ENTROPY_BETA')
gamma = config.getfloat('TRAINING', 'DISCOUNT_RATE')
lambd = 0.98

lr_a = 1e-3
lr_c = 1e-4

## Save/Summary
save_network_frequency = config.getint('TRAINING','SAVE_NETWORK_FREQ')
save_stat_frequency = config.getint('TRAINING','SAVE_STATISTICS_FREQ')*4
moving_average_step = config.getint('TRAINING','MOVING_AVERAGE_SIZE')

# Env Settings
action_space = 3
vision_range = 19 
keep_frame = 1
vision_dx, vision_dy = 2*vision_range+1, 2*vision_range+1
nchannel = 6 * keep_frame
in_size = [None, vision_dx, vision_dy, nchannel]
nenv = 2

global_rewards = MA(moving_average_step)
global_episode_rewards = MA(moving_average_step)
global_length = MA(moving_average_step)
global_succeed = MA(moving_average_step)
global_episodes = 0

##### Launch the session and create Graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_CAPACITY, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

sess = tf.Session(config=config)
progbar = tf.keras.utils.Progbar(total_episodes,interval=10)

global_step = tf.Variable(0, trainable=False, name='global_step')
global_step_next = tf.assign_add(global_step, nenv)
network = Network(in_size=in_size, action_size=action_space, scope='global', sess=sess)

saver = tf.train.Saver(max_to_keep=3, var_list=network.get_vars)
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    
ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Load Model : ", ckpt.model_checkpoint_path)
    iuv(sess)
else:
    sess.run(tf.global_variables_initializer())
    print("Initialized Variables")

saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes) # Initial save

##### Environment Setting
map_size = 20
#meta_red = policy.policy_scheduler.PolicyGen()
#policy_red = policy.policy_clone.Clone(meta_red.gen_action)
#meta_red = policy.policy_scheduler
meta_red = policy.roombaV2

def make_env(map_size, policy_red):
    return lambda: gym.make('cap-v0', map_size=map_size,
	config_path='setting1.ini')

envs = [make_env(map_size, meta_red) for i in range(nenv)]
envs = SubprocVecEnv(envs, keep_frame)
num_blue = len(envs.get_team_blue())//nenv

def record( item, writer):
    summary = tf.Summary()
    for key, value in item.items():
        summary.value.add(tag=key, simple_value=value)
    writer.add_summary(summary,global_episodes)
    writer.flush()

def train(trajs, bootstrap=0.0, epoch=3, batch_size=64, writer=None, log=False, global_episodes=None):
    def batch_iter(batch_size, states, actions, logits, tdtargets, advantages):
        size = len(states)
        for _ in range(size // batch_size):
            rand_ids = np.random.randint(0, size, batch_size)
            yield states[rand_ids, :], actions[rand_ids], logits[rand_ids], tdtargets[rand_ids], advantages[rand_ids]

    buffer_s, buffer_a, buffer_tdtarget, buffer_adv, buffer_logit = [], [], [], [], []
    for idx, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        observations = traj[0]
        actions = traj[1]

        td_target, advantages = gae(traj[2], traj[3], bootstrap[idx],
                gamma, lambd, normalize=False)
        
        buffer_s.extend(observations)
        buffer_a.extend(actions)
        buffer_tdtarget.extend(td_target)
        buffer_adv.extend(advantages)
        buffer_logit.extend(traj[4])

    buffer_size = len(buffer_s)
    if buffer_size < 10:
        return

    for _ in range(epoch):
        for state, action, old_logit, tdtarget, advantage in  \
            batch_iter(batch_size, np.stack(buffer_s), np.stack(buffer_a),
                    np.stack(buffer_logit), np.stack(buffer_tdtarget), np.stack(buffer_adv)):
            network.update_global(
                state, action, tdtarget, advantage, old_logit, global_episodes, writer, log)


##### Subpolicy
subpol1 = policy.policy_A3C.PolicyGen(
        model_dir='./model/golub_attacker_a3c',
        input_name='global/state:0',
        output_name='global/actor/Softmax:0',
        name='attacker'
    )
subpol2 = policy.policy_A3C.PolicyGen(
        model_dir='./model/golub_scout_a3c',
        input_name='global/state:0',
        output_name='global/actor/Softmax:0',
        name='scout'
    )
subpol3 = policy.policy_A3C.PolicyGen(
        model_dir='./model/golub_defense_a3c',
        input_name='global/state:0',
        output_name='global/actor/Softmax:0',
        name='defense'
    )
subpol = [subpol1, subpol2, subpol3]
def get_action(states):
    o1, v1, logits1 = network.run_network(states)

    actions = []
    for opt, agent, state in zip(o1, envs.get_team_blue(), states):
        action = subpol[opt].gen_action([agent], state[np.newaxis,:], centered_obs=True)[0]
        actions.append(action)

    return o1, v1, logits1, np.array(actions)

##### Run
print('Training Initiated:')
global_episodes = sess.run(global_step) # Reset the counter
while global_episodes < total_episodes:
    log_on = global_episodes % save_stat_frequency == 0 and global_episodes != 0
    save_on = global_episodes % save_network_frequency == 0 and global_episodes != 0
    red_update = global_episodes % 2048 == 0 and global_episodes != 0
    
    # initialize parameters 
    episode_rew = np.zeros(nenv)
    prev_rew = np.zeros(nenv)
    was_alive = [True for agent in envs.get_team_blue()]
    was_done = [False for env in range(nenv)]

    trajs = [Trajectory(depth=5) for _ in range(num_blue*nenv)]
    
    # Bootstrap
    s1 = envs.reset()
    a1, v1, logits1, a_sub = get_action(s1)

    # Rollout
    stime = time.time()
    for step in range(max_ep+1):
        s0 = s1
        a, v0 = a1, v1
        logits = logits1
        
        a_reshape = np.reshape(a_sub, [nenv, num_blue])
        s1, raw_reward, done, info = envs.step(a_reshape)
        is_alive = [agent.isAlive for agent in envs.get_team_blue()]
        reward = (raw_reward - prev_rew - 0.1*step)

        if step == max_ep:
            reward[:] = -100
            done[:] = True

        reward /= 100.0
        episode_rew += reward

    
        a1, v1, logits1, a_sub = get_action(s1)
        #a1[is_alive], v1[is_alive], logits1[is_alive] = network.run_network(s1[is_alive])
        for idx, d in enumerate(done):
            if d:
                v1[idx*num_blue: (idx+1)*num_blue] = 0.0

        # push to buffer
        for idx, agent in enumerate(envs.get_team_blue()):
            env_idx = idx // num_blue
            if was_alive[idx] and not was_done[env_idx]:
                trajs[idx].append([s0[idx], a[idx], reward[env_idx], v0[idx], logits[idx]])

        prev_rew = raw_reward
        was_alive = is_alive
        was_done = done

        if np.all(done):
            break
            
    stime = time.time()
    train(trajs, v1, 2, 64, writer, log_on, global_episodes)

    steps = []
    for env_id in range(nenv):
        steps.append(max([len(traj) for traj in trajs[env_id*num_blue:(env_id+1)*num_blue]]))
    global_episode_rewards.append(np.mean(episode_rew))
    global_rewards.append(np.mean(raw_reward))
    global_length.append(np.mean(steps))
    global_succeed.append(np.mean(envs.blue_win()))

    global_episodes += nenv
    sess.run(global_step_next)
    progbar.update(global_episodes)

    if log_on:
        record({
            'Records/mean_reward': global_rewards(),
            'Records/mean_length': global_length(),
            'Records/mean_succeed': global_succeed(),
            'Records/mean_episode_reward': global_episode_rewards(),
        }, writer)
        
    if save_on:
        saver.save(sess, MODEL_PATH+'/ctf_policy.ckpt', global_step=global_episodes)

    #if red_update:
        #meta_red.reset_network_weight()