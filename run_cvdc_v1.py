import pickle

import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import shutil
import argparse
import configparser

import signal
import threading
import multiprocessing

import tensorflow as tf

import time
import gym
import numpy as np
import random
import math
from collections import defaultdict
from functools import partial

from smac.env import StarCraft2Env
from SC2Wrappers import SMACWrapper, FrameStacking

from utility.utils import MovingAverage
from utility.utils import interval_flag, path_create
from utility.buffer import Trajectory
from utility.buffer import expense_batch_sampling as batch_sampler
from utility.logger import *
from utility.gae import gae

# from utility.slack import SlackAssist

from method.CVDC_ import SF_CVDC as Network

parser = argparse.ArgumentParser(description="CVDC(learnability) PredPrey")
parser.add_argument("--train_number", type=int, help="training train_number")
parser.add_argument("--machine", type=str, help="training machine")
parser.add_argument("--map", type=str, default='8m', help='the map of the game')
parser.add_argument("--silence", action="store_false", help="call to disable the progress bar")
parser.add_argument("--print", action="store_true", help="print out the progress in detail")
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--training_episodes", type=int, default=10000000, help='number of training episodes')
parser.add_argument("--gpu", action="store_false", help='Use of the GPU')
parser.add_argument("--gamma", type=float, default=0.99, help='gamma')
parser.add_argument("--lr", type=float, default=1E-4, help='lr')
parser.add_argument("--clr", type=float, default=1E-4, help='clr')
parser.add_argument("--ent", type=float, default=0.0, help='entropyBeta')
parser.add_argument("--rew", type=float, default=0.5, help='reward_beta')
parser.add_argument("--dec", type=float, default=1.0, help='decoder_beta')
parser.add_argument("--psi", type=float, default=0.1, help='psi_beta')
parser.add_argument("--crit", type=float, default=0.05, help='critic_beta')
parser.add_argument("--q", type=float, default=0.05, help='q_beta')
parser.add_argument("--learn", type=float, default=0.01, help='learnability_beta')
parser.add_argument("--epoch", type=int, default=1, help='epoch')
parser.add_argument("--bs", type=int, default=1024, help='buffer_size')
parser.add_argument("--mbs", type=int, default=128, help='minibatch_size')
parser.add_argument("--frames", type=int, default=4, help='frames')
parser.add_argument("--lstm", type=str,default="LSTM", help='LSTM Type')
parser.add_argument("--single", type=bool,default=False, help='Single Shared Network...')
args = parser.parse_args()

if args.gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PROGBAR = args.silence
PRINT = args.print

## Training Directory Reset
TRAIN_NAME = "CVDC_SC2_{}_{}_{:02d}".format(
    args.machine,
    args.map,
    args.train_number,
)
#slack_assist = SlackAssist(training_name=TRAIN_NAME, channel_name="#nodes")
TRAIN_TAG = "CVDC(learnability), " + TRAIN_NAME
LOG_PATH = "./logs/" + TRAIN_NAME
MODEL_PATH = "./model/" + TRAIN_NAME
SAVE_PATH = "./save/" + TRAIN_NAME
GPU_CAPACITY = 0.95

## Data Path
path_create(LOG_PATH)
path_create(MODEL_PATH)
path_create(SAVE_PATH)

# Training
total_episodes = args.training_episodes
gamma = args.gamma  # GAE - discount
lambd = 0.95  # GAE - lambda

# Log
test_interval = 2000
save_network_frequency = 4096
save_stat_frequency = 2000
save_image_frequency = 2000
moving_average_step = 2000 # MA for recording episode statistics

## Environment
frame_stack = args.frames
env = StarCraft2Env(
    map_name=args.map,
    # reward_scale_rate=20,
    replay_dir=SAVE_PATH
)


env_info = env.get_env_info()
# env = SMACWrapper(env)
# env = FrameStacking(env, numFrames=frame_stack, lstm=True)
# Environment/Policy Settings
print(env_info)
env.reset()
done=False
while not done:
    avail_actions = env.get_avail_actions()
    sampleAction = []
    for avail_action_i in avail_actions:
        sampleAction.append(np.random.choice(np.where(np.asarray(avail_action_i)==1)[0]))
    r,done,_=env.step(sampleAction)
agent_types = []
for i in range(env_info["n_agents"]):
    agent_types.append(env.agents[i].unit_type)
num_agent_types = len(set(agent_types))
agent_types_assign = list(set(agent_types))
print(agent_types)

# print(type(env.agents[1]))
# print(dir(env.agents[1]))
# exit()
action_space = env_info["n_actions"]
num_agent = env_info["n_agents"]
state_shape = [frame_stack, env_info["state_shape"]]#env_info["state_shape"] * frame_stack
obs_shape = [frame_stack, env_info["obs_shape"]+num_agent+action_space]#env_info["obs_shape"] * frame_stack
episode_limit = env_info["episode_limit"]
# exit()
## Batch Replay Settings
minibatch_size = args.mbs
epoch = args.epoch
buffer_size = args.bs
drop_remainder = False

## Logger Initialization
log_episodic_reward = MovingAverage(100)
log_winrate = MovingAverage(100)
log_looptime = MovingAverage(moving_average_step)
log_traintime = MovingAverage(moving_average_step)

## Progress bar
if PROGBAR:
    progbar = tf.keras.utils.Progbar(None, unit_name=TRAIN_TAG)

# Network Setup
atoms = 128
network = Network(
    state_shape=state_shape,
    obs_shape=obs_shape,
    action_space=action_space,
    atoms=atoms,
    save_path=MODEL_PATH,
    agent_types=agent_types,
    lr=args.lr,
    clr=args.clr,
    entropy=args.ent,
    reward_beta=args.rew,
    decoder_beta=args.dec,
    psi_beta=args.psi,
    critic_beta=args.crit,
    q_beta=args.q,
    learnability_beta=args.learn,
    network_type=args.lstm,
    single=args.single,
)
global_episodes = network.initiate()
print(global_episodes)
writer = tf.summary.create_file_writer(LOG_PATH)

class Stacked_state:
    def __init__(self, keep_frame, axis,lstm=False):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm=lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame

    def __call__(self, obj=None):
        if obj is None:
            if self.lstm:
                # print(np.stack(self.stack, axis=self.axis).shape)
                return np.stack(self.stack, axis=self.axis)
            else:
                return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        if self.lstm:
            # print(np.stack(self.stack, axis=self.axis).shape)
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)

# TRAINING
def train_central(
    network,
    trajs,
    bootstrap=0.0,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    traj_buffer = defaultdict(list)
    for idx, traj in enumerate(trajs):
        # Forward
        states = np.array(traj[0])
        last_state = np.array(traj[3])[-1:, ...]
        reward = traj[1]

        env_critic, _ = network.run_network_central(states)
        _env_critic, _ = network.run_network_central(last_state)
        critic = env_critic["critic"].numpy().tolist()
        _critic = _env_critic["critic"].numpy()

        # TD-target
        td_target_c, _ = gae(reward, critic, _critic, gamma, lambd,
            normalize=False,
            td_lambda=True
        )

        traj_buffer["state"].extend(traj[0])
        traj_buffer["td_target_c"].extend(td_target_c)
        traj_buffer["old_value"].extend(critic)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            {
                "state": np.stack(traj_buffer["state"]).astype(np.float32),
                "td_target_c": np.stack(traj_buffer["td_target_c"]).astype(np.float32),
                "old_value": np.stack(traj_buffer["old_value"]).astype(np.float32),
            }
        )
        .shuffle(64)
        .batch(batch_size, drop_remainder=True)
    )

    network.update_central(
        train_dataset, epoch, writer=writer, log=log, step=step, tag="losses/"
    )

def train_decentral(
    agent_trajs,
    epoch=epoch,
    batch_size=minibatch_size,
    writer=None,
    log=False,
    step=None,
):
    # Agent trajectory processing
    traj_buffer = [defaultdict(list) for i in range(num_agent_types)]
    advantage_lists = []
    f1_list = []
    f2_list = []
    fc_list = []
    for i,traj in enumerate(agent_trajs):
        reward = traj[2]
        mask = traj[3]
        critic = traj[5]
        phi = traj[7]
        psi = traj[8]
        _critic = traj[9][-1]
        _psi = traj[10][-1]

        cent_state = np.array(traj[14])
        env_critic, _ = network.run_network_central(cent_state)
        env_critic = env_critic["critic"].numpy().tolist()
        cent_last_state = np.array(traj[15])[-1:, ...]
        _env_critic, _ = network.run_network_central(cent_last_state)
        _env_critic = _env_critic["critic"].numpy()

        icritic = traj[12]
        mask = traj[3]

        '''
        dc = np.array(critic[1:])-np.array(critic[:-1])
        dc1 = np.array(env_critic[1:])-np.array(env_critic[:-1])
        dc2 = np.array(icritic[1:])-np.array(icritic[:-1])
        f1_list.append(np.mean((dc * dc1)>0))
        f2_list.append(np.mean((dc * dc2)>0))
        fc_list.append(np.mean((dc * dc)>0))
        '''

        # Zero bootstrap because all trajectory terminates
        td_target_c, advantages_global = gae(
            reward, critic, _critic,
            gamma, lambd, normalize=False,
            td_lambda=True, # mask=mask,
            standardize_td=True,
        )
        _, advantages_global = gae(
            reward,
            env_critic,
            _env_critic,
            gamma,
            lambd,
            normalize=False,
            td_lambda=True,
            #mask=mask,
        )
        _, advantages = gae(
            traj[11],
            traj[12],
            traj[13][-1],
            gamma,
            lambd,
           # mask=mask,
            normalize=False,
        )
        td_target_psi, _ = gae(
            phi,
            psi,
            _psi,  # np.zeros_like(phi[0]),
            gamma,
            lambd,
           # mask=np.array(mask)[:, None],
            discount_adv=False,
            normalize=False,
            td_lambda=False,
        )

        '''
        traj_length = np.where(mask)[0]
        if len(traj_length) == 0:
            traj_length = len(traj[0])
        else:
            traj_length = traj_length[0]
        '''
        j = agent_types_assign.index(agent_types[i%num_agent])
        traj_buffer[j]["state"].extend(traj[0])
        traj_buffer[j]["next_state"].extend(traj[4])
        traj_buffer[j]["log_logit"].extend(traj[6])
        traj_buffer[j]["action"].extend(traj[1])
        traj_buffer[j]["old_value"].extend(critic)
        traj_buffer[j]["td_target_psi"].extend(td_target_psi)
        traj_buffer[j]["advantage"].extend(advantages_global)
        traj_buffer[j]["td_target_c"].extend(td_target_c)
        traj_buffer[j]["rewards"].extend(reward)
        traj_buffer[j]["avail_actions"].extend(traj[16])

    train_datasets = []  # Each for type of agents
    for atype in range(num_agent_types):
        #traj_buffer = traj_buffer_list[atype]

        # Normalize Advantage (global)
        _adv = np.array(traj_buffer[atype]["advantage"]).astype(np.float32)
        _adv = (_adv - _adv.mean()) / (_adv.std()+1e-9)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                {
                    "state": np.stack(traj_buffer[atype]["state"]).astype(np.float32),
                    "old_log_logit": np.stack(traj_buffer[atype]["log_logit"]).astype(np.float32),
                    "action": np.stack(traj_buffer[atype]["action"]),
                    "old_value": np.stack(traj_buffer[atype]["old_value"]).astype(np.float32),
                    "td_target": np.stack(traj_buffer[atype]["td_target_psi"]).astype(np.float32),
                    "advantage": _adv,
                    "td_target_c": np.stack(traj_buffer[atype]["td_target_c"]).astype(np.float32),
                    "rewards": np.stack(traj_buffer[atype]["rewards"]).astype(np.float32),
                    "next_state": np.stack(traj_buffer[atype]["next_state"]).astype(np.float32),
                    "avail_actions": np.stack(traj_buffer[atype]["avail_actions"]).astype(np.bool)
                }
            )
            .shuffle(64)
            .batch(batch_size, drop_remainder=drop_remainder)
        )
        train_datasets.append(train_dataset)

    network.update_decentral(
        train_datasets, epoch, writer=writer, log=log, step=step, tag="losses/",
    )
    if log:
        with writer.as_default():
            tag = "advantages/"
            for idx, adv_list in enumerate(advantage_lists):
                tb_log_histogram(
                    np.array(adv_list), tag + "dec_advantages_{}".format(idx), step=step
                )
            #tf.summary.scalar('factoredness/f1', np.mean(f1_list), step=step)
            #tf.summary.scalar('factoredness/f2', np.mean(f2_list), step=step)
            #tf.summary.scalar('factoredness/fc', np.mean(fc_list), step=step)
            writer.flush()


def run_network(observations, env,greedy=False):
    # Get available actions

    # Run decentral network
    observation_lists = [[] for i in range(num_agent_types)]
    avail_actions_lists = [[] for i in range(num_agent_types)]
    avail_actions_array = []
    for i,observation_i in enumerate(observations):
        avail_action = env.get_avail_agent_actions(i)
        type_assign = agent_types_assign.index(agent_types[i])
        avail_actions_lists[type_assign].append(avail_action)
        observation_lists[type_assign].append(observation_i)
        avail_actions_array.append(avail_action)

    observations = [np.array(observation_i) for observation_i in observation_lists]
    avail_actions = [np.array(avail_actions_i) for avail_actions_i in avail_actions_lists]

    actor_out, critic_out = network.run_network_decentral(observations, avail_actions)

    actor = {}
    for i in range(num_agent_types):
        # print(actor_out[i].keys())
        for key,value in actor_out[i].items():
            if i == 0:
                actor[key]=value
            else:
                actor[key] = tf.concat([actor[key],value],axis=0)

    critic = {}
    for i in range(num_agent_types):
        for key,value in critic_out[i].items():
            if i == 0:
                critic[key]=value
            else:
                critic[key] = tf.concat([critic[key],value],axis=0)


    # dec_results[0] --> actor. actor['log_softmax'] --> tf.random.categorical
    # Get action
    probs = actor['softmax'].numpy().squeeze()
    action_probs = probs * avail_actions_array
    try:
        if greedy:
            a1 = []
            for idx, p in enumerate(action_probs):
                if np.isclose(p.sum(), 0):
                    avail_actions = env.get_avail_agent_actions(idx)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
                    a1.append(action)
                else:
                    p = p/p.sum()
                    action = np.argmax(p)
                    a1.append(action)
        else:
            a1 = []
            for idx, p in enumerate(action_probs):
                if np.isclose(p.sum(), 0):
                    avail_actions = env.get_avail_agent_actions(idx)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
                    a1.append(action)
                else:
                    p = p/p.sum()
                    action = np.random.choice(action_space, p=p)
                    a1.append(action)
    except ValueError:
        print(probs)
        print(action_probs)
        print(observations)
        raise ValueError

    # Container
    '''
    a1 = np.empty([num_agent], dtype=np.int32)
    vg1 = np.empty([num_agent], dtype=np.float32)
    vc1 = np.empty([num_agent], dtype=np.float32)
    phi1 = np.empty([num_agent, atoms], dtype=np.float32)
    psi1 = np.empty([num_agent, atoms], dtype=np.float32)
    log_logits1 = np.empty([num_agent, action_space], dtype=np.float32)
    reward_pred1 = np.empty([num_agent], dtype=np.float32)
    '''

    # Postprocessing
    '''
    for (a, actor, critic), mask in zip(results, agent_type_masking):
        vg = critic["critic"]
        vc = critic["icritic"]
        phi = critic["phi"]
        psi = critic["psi"]
        log_logits = actor["log_softmax"]
        reward_pred = critic["reward_predict"]

        a1[mask] = a
        vg1[mask] = vg
        vc1[mask] = vc
        phi1[mask, :] = phi
        psi1[mask, :] = psi
        log_logits1[mask, :] = log_logits
        reward_pred1[mask] = reward_pred
    action = np.reshape(a1, [NENV, num_blue])
    '''
    vg1 = critic["critic"]
    vc1 = critic["icritic"]
    phi1 = critic["phi"]
    psi1 = critic["psi"]
    log_logits1 = actor["log_softmax"]
    reward_pred1 = critic["reward_predict"]

    return a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, avail_actions_array


def TestNetwork(env,episodes):
    stackedStates_obs_test = Stacked_state(frame_stack, 1,True)
    episode_rewards=[]
    episode_win=[]
    for i in range(episodes):
        env.reset()
        terminated=False
        stackedStates_obs_test.initiate(np.vstack(env.get_obs()))
        o1_=np.vstack(env.get_obs())
        o2_=np.concatenate([o1_,np.eye(num_agent),np.zeros((num_agent,action_space))],axis=1)
        stackedStates_obs.initiate(o2_)
        o1 = stackedStates_obs()
        episode_reward=0
        while not terminated:
            a1, _, _, _, _, _, _, _ = run_network(o1, env,greedy=True)
            reward, terminated, info = env.step(a1)
            o1_=np.vstack(env.get_obs())
            oh = np.zeros((np.asarray(a0).size, action_space))
            oh[np.arange(np.asarray(a0).size),np.asarray(a0)] = 1
            o2_=np.concatenate([o1_,np.eye(num_agent),oh],axis=1)
            o1 = stackedStates_obs(o2_)
            episode_reward += reward
            if "battle_won" not in info:
                info["battle_won"]=False
        episode_rewards.append(episode_reward)
        episode_win.append(float(info["battle_won"]))
    return np.mean(episode_rewards), np.mean(episode_win)


stackedStates_obs = Stacked_state(frame_stack, 1,True)

stackedStates_states = Stacked_state(frame_stack, 1,True)

while global_episodes < total_episodes:
#while True:
    batch = []
    batch_size = 0
    dec_batch = []
    dec_batch_size = 0
    while dec_batch_size < buffer_size:
        # initialize parameters
        dec_trajs = [Trajectory(depth=17) for _ in range(num_agent)]
        cent_trajs = Trajectory(depth=4)

        # Reset Game
        env.reset()
        done = np.array([False]*num_agent)
        episode_reward = 0

        # Bootstrap
        stackedStates_states.initiate(np.expand_dims(env.get_state().astype(np.float32),0))
        s1 = stackedStates_states()

        o1_=np.vstack(env.get_obs())
        o2_=np.concatenate([o1_,np.eye(num_agent),np.zeros((num_agent,action_space))],axis=1)
        stackedStates_obs.initiate(o2_)
        o1 = stackedStates_obs()

        a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, _avail_actions = run_network(o1, env)

        # Rollout
        stime_roll = time.time()
        step = 0
        while not done.all():
            s0 = s1
            o0 = o1
            a0 = a1
            vg0 = vg1
            vc0 = vc1
            psi0 = psi1
            phi0 = phi1
            log_logits0 = log_logits1
            was_done = done
            avail_actions = _avail_actions

            # Run environment
            #reward, done, _ = env.step(a0)
            # print(a0)
            reward, terminated, info = env.step(a0)

            s1 = stackedStates_states(np.expand_dims(env.get_state().astype(np.float32),0))
            # o1 = stackedStates_obs(np.vstack(env.get_obs()))
            o1_=np.vstack(env.get_obs())
            oh = np.zeros((np.asarray(a0).size, action_space))
            oh[np.arange(np.asarray(a0).size),np.asarray(a0)] = 1
            o2_=np.concatenate([o1_,np.eye(num_agent),oh],axis=1)
            o1 = stackedStates_obs(o2_)

            step += 1
            episode_reward += reward

            # Run network
            a1, vg1, vc1, phi1, psi1, log_logits1, reward_pred1, _avail_actions = run_network(o1, env)
            # validAction = self.env.get_avail_actions()
            if terminated:
                done = np.asarray([terminated]*env.n_agents)
            else:
                # done=[]
                # for validAction in _avail_actions:
                #     if validAction[0] == 1:
                #         done.append( True)
                #     else:
                #         done.append(False)
                # done = np.asarray(done)
                done = np.asarray([terminated]*env.n_agents)

            if "battle_won" not in info:
                info["battle_won"]=False

            # Experience recording
            for idx in range(num_agent):
                if np.all(o0[idx]==0):
                    continue
                dec_trajs[idx].append(
                    [
                        o0[idx],
                        a0[idx],
                        reward,  # + reward_pred1[idx], # Advantage
                        was_done[idx],  # masking
                        o1[idx],
                        vg0[idx],  # Advantage
                        log_logits0[idx],  # PPO
                        phi0[idx].numpy(),  # phi: one-step ahead
                        psi0[idx].numpy(),
                        vg1[idx],
                        psi1[idx].numpy(),
                        reward,
                        #reward[env_idx]-(reward_pred1[idx] if reward[env_idx] else 0),
                        #reward[env_idx]-reward_pred1[idx],
                        vc0[idx],
                        vc1[idx],
                        s0[0],
                        s1[0],
                        avail_actions[idx]
                    ]
                )
            cent_trajs.append([
                s0[0],
                reward,
                was_done.all(),
                s1[0],
            ])
        etime_roll = time.time()



        log_episodic_reward.append(episode_reward)
        log_looptime.append(etime_roll - stime_roll)
        log_winrate.append(info["battle_won"])

        # Buffer
        dec_batch.extend(dec_trajs)
        dec_batch_size += sum([len(traj) for traj in dec_trajs])
        batch.append(cent_trajs)
        batch_size += step

        if PRINT:
            print("Reward in episode {} = {} {}".format(global_episodes, episode_reward, TRAIN_NAME))

        # Stepper
        global_episodes += step
        if PROGBAR:
            progbar.update(global_episodes)

    # decentralize training
    if PRINT:
        print('training: {}'.format(dec_batch_size))
    stime_train = time.time()
    log = interval_flag(global_episodes, save_image_frequency, "im_log")
    train_decentral(
        dec_batch,
        epoch=epoch,
        batch_size=minibatch_size,
        writer=writer,
        log=log,
        step=global_episodes,
    )
    etime_train = time.time()
    log_traintime.append(etime_train - stime_train)

    # centralize training
    if PRINT:
        print('training(central): {}'.format(batch_size))
    log_tc_on = interval_flag(global_episodes, save_image_frequency, 'tc_log')
    train_central(
        network,
        batch,
        0,
        epoch,
        minibatch_size,
        writer,
        log_tc_on,
        global_episodes
    )

    # Log
    test_on = interval_flag(global_episodes, test_interval, "test")
    if test_on:
        r_test,wr_test=TestNetwork(env,16)
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "test_env_reward", r_test, step=global_episodes)
            tf.summary.scalar(tag + "test_winrate", wr_test, step=global_episodes)
            writer.flush()

    log_on = interval_flag(global_episodes, save_stat_frequency, "log")
    if log_on:
        with writer.as_default():
            tag = "baseline_training/"
            tf.summary.scalar(tag + "env_reward", log_episodic_reward(), step=global_episodes)
            tf.summary.scalar(tag + "winrate", log_winrate(), step=global_episodes)
            tf.summary.scalar(tag + "rollout_time", log_looptime(), step=global_episodes)
            tf.summary.scalar(tag + "train_time", log_traintime(), step=global_episodes)
            writer.flush()

    # Network Save
    save_on = interval_flag(global_episodes, save_network_frequency, "save")
    if save_on:
        network.save(global_episodes)

    # Save Gameplay
    log_save_analysis = interval_flag(global_episodes, 1024 * 4, "save_log")
    if log_save_analysis:
        pass
