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

# Scoreboard
elo = Elo()
elo.load()

# Read Possible Player
players = []
def read_player(fpath):
    players = []
    with open(fpath, 'r') as f:
        for line in f:
            model_path, step, group_name = line.strip().split(',')
            step = int(step)
            players.append((model_path, step, group_name))
    return players
players = read_player('competition_player.txt')

for _,_,name in players:
    if not elo.contains(name):
        elo.addPlayer(name)
for name in ['Roomba', 'Zeros', 'Random']:
    if not elo.contains(name):
        elo.addPlayer(name)

# Set Config
N = 5 # Length of set
win_cutoff = 0.5

try:
    nn_policy_1 = policy.UAV()
    nn_policy_2 = policy.UAV()
    basic_policies = {'Roomba': policy.Roomba(), 'Zeros': policy.Zeros(), 'Random': policy.Random()}
    basic_policies_name = ['Roomba', 'Zeros', 'Random']
    episode = 0
    player1, player2 = 'Roomba', 'Zeros'
    policy1, policy2 = [], []

    resetCount = 0
    for _ in range(100):
        ## Reread Players
        #if episode % 100 == 0:
        #    players = read_player('competition_player.txt')
        #    for _,_,name in players:
        #        if not elo.contains(name):
        #            elo.addPlayer(name)

        # Player Selection
        p1, p2 = random.sample(players + basic_policies_name, 2)
        p1_name = p1 if p1 in basic_policies_name else p1[2]
        p2_name = p2 if p2 in basic_policies_name else p2[2]
        if abs(elo.getPlayerRating(p1_name)-elo.getPlayerRating(p2_name)) > 471.0+resetCount*0.1: # Resample
            resetCount += 1
            continue
        else:
            resetCount = 0
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

        # Set Match
        match_results = []
        for _ in range(N):
            observation = env.reset(
                    map_size=20,
                    config_path='uav_settings.ini',
                    custom_board=random.choice(map_paths),
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
                match_results.append(0)
            elif env.blue_win:
                match_results.append(1)
            elif env.red_win:
                match_results.append(-1)
            else: 
                match_results.append(0)
        
        # Record Set Match
        result = sum(match_results)
        if result > 0:
            elo.recordMatch(player1, player2, winner=player1, verbose=1)
        elif result < 0:
            elo.recordMatch(player1, player2, winner=player2, verbose=1)
        else:
            elo.recordMatch(player1, player2, draw=True, verbose=1)

        episode += 1

    # Draw Table
    elo_string = str(elo)
    with open('competition_result.txt', 'w') as f:
        print(elo_string)
        f.write(elo_string)
    elo.save()

except KeyboardInterrupt:
    env.close()
    del gym.envs.registry.env_specs['cap-v0']

    elo_string = str(elo)
    with open('competition_result.txt', 'w') as f:
        print(elo_string)
        f.write(elo_string)
    elo.save()
    print("CtF environment Closed")

