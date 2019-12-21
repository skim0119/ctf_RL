import os
from os.path import isfile, join
import random

import time
import gym
import gym_cap
import numpy as np

# the modules that you can use to generate the policy.
import policy

from utility.elopy import Elo

env = gym.make("cap-v0")

# Fair map list
map_path = 'fair_uav'
map_paths = [join(map_path,f) for f in os.listdir(map_path) if isfile(join(map_path, f))]

# Read Possible Player
def read_player(fpath):
    players = []
    with open(fpath, 'r') as f:
        for line in f:
            model_path, step, group_name = line.strip().split(',')
            step = int(step)
            players.append((model_path, step, group_name))
    return players
players = read_player('competition_player.txt')

# Scoreboard
elo = Elo()
for _,_,name in players:
    if not elo.contains(name):
        elo.addPlayer(name)
elo.addPlayer('Roomba')
elo.addPlayer('Zeros')
elo.addPlayer('Random')

try:
    nn_policy_1 = policy.UAV()
    nn_policy_2 = policy.UAV()
    basic_policies = {'Roomba': policy.Roomba(), 'Zeros': policy.Zeros(), 'Random': policy.Random()}
    basic_policies_name = ['Roomba', 'Zeros', 'Random']
    episode = 0
    player1, player2 = 'Roomba', 'Zeros'
    policy1, policy2 = [], []
    while True:
        # Player Selection
        if episode % 5 == 0:
            p1, p2 = random.sample(players + basic_policies_name, 2)
            if p1 in basic_policies_name:
                player1 = p1
                policy1 = basic_policies[player1]
            else:
                player1 = p1[2]
                policy1 = nn_policy_1
                policy1.reload_network(path=p1[0], step=p1[1])
            if p2 in basic_policies_name:
                player2 = p2
                policy2 = basic_policies[player2]
            else:
                player2 = p2[2]
                policy2 = nn_policy_2
                policy2.reload_network(path=p2[0], step=p2[1])

        observation = env.reset(
                map_size=20,
                config_path='uav_settings.ini',
                policy_blue=policy1,
                policy_red=policy2,
            )
        env.CONTROL_ALL = False
        t = 0
        done = False
        while not done:
            observation, reward, done, info = env.step()  # feedback from environment
            t += 1
            if t == 150:
                break

        if env.red_win and env.blue_win:
            elo.recordMatch(player1, player2, draw=True)
        elif env.blue_win:
            elo.recordMatch(player1, player2, winner=player1)
        elif env.red_win:
            elo.recordMatch(player1, player2, winner=player2)
        else: 
            elo.recordMatch(player1, player2, draw=True)

        episode += 1

        # Draw Table
        if episode % 100 == 0:
            with open('competition_result.txt', 'w') as f:
                print('played_episode = ', episode)
                tb = elo.getRatingList()
                tb = sorted(tb, key=lambda x: x[1])[::-1]
                format_string = '|{:>15}|{:>5}|'
                print(format_string.format('Name', 'Rate'))
                print('-'*15)
                for name, rate in tb:
                    rate = int(rate)
                    print(format_string.format(name, rate))
                    f.write(format_string.format(name, rate))

except KeyboardInterrupt:
    env.close()
    del gym.envs.registry.env_specs['cap-v0']

    print("CtF environment Closed")


