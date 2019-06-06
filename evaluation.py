"""
This module is written to run simple evaluation between policies.

It will generate a win-loss rate and general statistics on the game.


TODO:
    - Include Rendering
    - Include death-rate statistics
"""

import os

import time
import gym
import gym_cap
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

# the modules that you can use to generate the policy.
import policy.random
import policy.roomba
import policy.zeros
import policy.policy_A3C
import policy.policy_ensemble

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import inquirer
import shutil
import re
from tqdm import tqdm

MAX_LENGTH = 150

BUILT_POLICIES = ['roomba', 'random', 'zeros', 'A3C', 'A3C_ensemble']
POLICIES = [policy.roomba, policy.random, policy.zeros, policy.policy_A3C, policy.policy_ensemble]

questions = [
    inquirer.List(
        'red_policy',
        message="Which policy for RED?",
        choices=BUILT_POLICIES,
    ),
    inquirer.List(
        'blue_policy',
        message="Which policy for BLUE?",
        choices=BUILT_POLICIES,
    ),
]
answers = inquirer.prompt(questions)

policy_red = POLICIES[BUILT_POLICIES.index(answers['red_policy'])]
policy_blue = POLICIES[BUILT_POLICIES.index(answers['blue_policy'])]

# Environment
env = gym.make("cap-v0").unwrapped  # initialize the environment
ls_a3c = ['default'] + os.listdir('model')
# Set Red Policy
if answers['red_policy'] == 'A3C':
    questions = [
        inquirer.List(
            'path',
            message="Which policy for RED?",
            choices=ls_a3c,
        ),
    ]
    a3c_choice = inquirer.prompt(questions)
    if a3c_choice['path'] == 'default':
        policy_red = policy_red.PolicyGen(color='red')
    else:
        policy_red = policy_red.PolicyGen(
            model_dir='./model/' + a3c_choice['path'],
            color='red'
        )
else:
    policy_red = policy_red.PolicyGen(env.get_map, env.get_team_red)

# Set Blue Policy
if answers['blue_policy'] == 'A3C':
    questions = [
        inquirer.List(
            'path',
            message="Which policy for RED?",
            choices=ls_a3c,
        ),
    ]
    a3c_choice = inquirer.prompt(questions)
    if a3c_choice['path'] == 'default':
        policy_blue = policy_blue.PolicyGen()
    else:
        policy_blue = policy_blue.PolicyGen(model_dir='./model/' + a3c_choice['path'])
else:
    policy_blue = policy_blue.PolicyGen(env.get_map, env.get_team_blue)


print('----Environment is Ready----')

questions = [
    inquirer.Text(
        'run_episode',
        message="How many episode to run? (Positive Number)",
        validate=lambda _, x: re.match('^[+]?\d+([.]\d+)?$', x),
        default='2000',
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
    inquirer.Text(
        'run_name',
        message="Run Name? (new eval/'name' directory)",
    )
]
params = inquirer.prompt(questions)
env.red_partial_visibility = params['run_partial_red'] == 'yes'
env.num_blue_ugv = int(params['num_blue'])
env.num_red_ugv = int(params['num_red'])
env.reset(map_size=int(params['map_size']), policy_blue=policy_blue, policy_red=policy_red)
env()

# Create save directory (eval/name)
path = "eval/" + params['run_name']
if os.path.exists(path):
    shutil.rmtree(path)
    print('Removed existing directory')
try:
    os.makedirs(path)
except OSError:
    raise OSError(f'Creation of the directory {path} failed')
else:
    print("Successfully created the directory %s" % path)


def play_episode():
    # Reset environmnet
    obs = env.reset()

    # Rollout episode
    episode_length = 0
    done = 0
    prev_reward = 0
    cumulative_reward = 0
    while (done == 0):
        episode_length += 1

        # state consists of the centered obss of each agent
        if params['run_partial_blue'] == 'yes':
            action = policy_blue.gen_action(env.get_team_blue, env.get_obs_blue)
        else:
            action = policy_blue.gen_action(env.get_team_blue, env._env)  # Full observability

        obs, env_reward, done, info = env.step(action)
        reward = (env_reward - prev_reward) / 100
        # stop the episode if it goes too long
        if episode_length > MAX_LENGTH:
            reward = -1.
            done = True
        cumulative_reward += reward
        prev_reward = env_reward

    # Post Statistics
    success_flag = env.blue_win
    survival_rate = sum([agent.isAlive for agent in env.get_team_blue]) / len(env.get_team_blue)
    kill_rate = sum([not agent.isAlive for agent in env.get_team_red]) / len(env.get_team_red)

    # Closer
    return episode_length, reward, survival_rate, kill_rate, success_flag


# Run
start_time = time.time()
total_episodes = int(params['run_episode'])
data_column = ['length', 'reward', 'survive', 'kill', 'win']
data = pd.DataFrame(columns=data_column)

for episode in tqdm(range(total_episodes)):
    data.loc[episode] = play_episode()
env.close()

# Post Statistics and Save
plt.ioff()
data = data.reset_index()
for name in data_column:
    mean = data[name].rolling(window=5).mean()
    std = data[name].rolling(window=5).std()
    data["moving_average"] = mean
    data["moving_std"] = std
    data["up"] = mean + std
    data["down"] = mean - std

    plt.figure()
    ax = sns.lineplot(x="index", y="moving_average", data=data)
    plt.title(name + f' (evaluation {data[name].mean()})')
    plt.xlabel('episode')
    plt.ylabel(name)
    #  ax.fill_between(x=data["index"], y1=data["down"], y2=data["up"], alpha=0.3)

    try:
        save_path = path + '/' + name + '.png'
        plt.savefig(save_path)
    except KeyError:
        raise KeyError('required is a Required Argument')

    print(f'{name} plot created: Average {name} = {data[name].mean()}')

print(f'Total Run Time : {time.time()-start_time} sec')
