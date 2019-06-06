%load_ext autoreload
%autoreload 2
import os

import time
import gym
import gym_cap
import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt
import tensorflow as tf

# the modules that you can use to generate the policy.
import policy.zeros
import policy.patrol # only for non-stochastic zone? 
import policy.random
import policy.roomba # Supposed to be heuristic
import policy.policy_RL # RL weights are in ./model directory
import policy.policy_RL_indv # RL weights are in ./model directory

# custom utilities
from utility.utils import MovingAverage as MA


env = gym.make("cap-v0") # initialize the environment

n_episode   = 500
ep_max      = 150
map_size    = 10
moving_ave  = 10
base_policy = [policy.zeros, policy.random]
base_nick = ['zero', 'random']

#policy_red  = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red ,model_dir=rl_model[0],color='red')
policy_blue  = policy.policy_RL_indv.PolicyGen(env.get_map, env.get_team_blue ,model_dir=rl_model[1],color='blue')
policy_red  = policy.zeros.PolicyGen(env.get_map, env.get_team_red)
'''policy_red = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red,
                                         model_dir=rl_model[0],
                                         color='red',
                                         input_name='global/global/state:0',
                                         output_name='global/global/actor/fully_connected_1/Softmax:0',
                                         import_scope='global'
                                        )'''
'''policy_blue = policy.policy_RL.PolicyGen(env.get_map, env.get_team_blue,
                                         model_dir=rl_model[0],
                                         color='blue',
                                         input_name='global/global/state:0',
                                         output_name='global/global/actor/fully_connected_1/Softmax:0',
                                         import_scope='global'
                                        )'''



blue_table = []
red_table  = []
draw_table = []
# reset the environment and select the policies for each of the team
s = env.reset(map_size=map_size,
            policy_red=policy_red,
            policy_blue=policy_blue)

progbar = tf.keras.utils.Progbar(n_episode)
for ep in range(n_episode+1):
    progbar.update(ep)
    total_reward = 0
    prev_reward = 0
    for frame in range(ep_max):
        action = policy_blue.gen_action(env.get_team_blue, env._env)
        s, reward, done, info = env.step(action)
        #s, reward, done, info = env.step()  # feedback from environment
        r = reward-prev_reward
        total_reward += r
        prev_reward = reward
        if done or frame==ep_max-1:
            if env.blue_win:
                blue_table.append((frame, total_reward))
            elif env.red_win:
                red_table.append((frame, total_reward))
            else:
                draw_table.append((frame, total_reward))
            break
    env.reset()
print(f'\nblue won: {len(blue_table)}')
print(f'red won: {len(red_table)}')
print(f'draw : {len(draw_table)}')

'''plt.figure()
plt.scatter(rand_jitter(length_table), rand_jitter(reward_table))
plt.title(f'red: {base_name}')
plt.legend(ctrl_nick+rl_nick)
plt.xlabel('episode length')
plt.ylabel('reward')'''

policy_red  = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red ,model_dir=rl_model[1],color='red')
#policy_blue  = policy.policy_RL_indv.PolicyGen(env.get_map, env.get_team_blue ,model_dir=rl_model[1],color='blue')
policy_blue  = policy.zeros.PolicyGen(env.get_map, env.get_team_blue)
'''policy_red = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red,
                                         model_dir=rl_model[0],
                                         color='red',
                                         input_name='global/global/state:0',
                                         output_name='global/global/actor/fully_connected_1/Softmax:0',
                                         import_scope='global'
                                        )'''
'''policy_blue = policy.policy_RL.PolicyGen(env.get_map, env.get_team_blue,
                                         model_dir=rl_model[0],
                                         color='blue',
                                         input_name='global/global/state:0',
                                         output_name='global/global/actor/fully_connected_1/Softmax:0',
                                         import_scope='global'
                                        )'''



blue_table = []
red_table  = []
draw_table = []
# reset the environment and select the policies for each of the team
s = env.reset(map_size=map_size,
            policy_red=policy_red,
            policy_blue=policy_blue)

progbar = tf.keras.utils.Progbar(n_episode)
for ep in range(n_episode+1):
    progbar.update(ep)
    total_reward = 0
    prev_reward = 0
    for frame in range(ep_max):
        action = policy_blue.gen_action(env.get_team_blue, env._env)
        s, reward, done, info = env.step(action)
        #s, reward, done, info = env.step()  # feedback from environment
        r = reward-prev_reward
        total_reward += r
        prev_reward = reward
        if done or frame==ep_max-1:
            if env.blue_win:
                blue_table.append((frame, total_reward))
            elif env.red_win:
                red_table.append((frame, total_reward))
            else:
                draw_table.append((frame, total_reward))
            break
    env.reset()
print(f'\nblue won: {len(blue_table)}')
print(f'red won: {len(red_table)}')
print(f'draw : {len(draw_table)}')

'''plt.figure()
plt.scatter(rand_jitter(length_table), rand_jitter(reward_table))
plt.title(f'red: {base_name}')
plt.legend(ctrl_nick+rl_nick)
plt.xlabel('episode length')
plt.ylabel('reward')'''