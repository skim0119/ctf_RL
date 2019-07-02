"""Patrolling agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

"""

import random

import numpy as np
import gym_cap.envs.const as const

from policy.policy import Policy
from policy.astar_flag import AStar

class Patrol(Policy):
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
        patrol: Private method to control a single unit.
    """

    def __init__(self):
        super().__init__()
        self.astar = AStar()
    
    def initiate(self, free_map, agent_list):
        self.team = agent_list[0].team
        other_team = const.TEAM2_BACKGROUND if self.team==const.TEAM1_BACKGROUND else const.TEAM1_BACKGROUND
        self.free_map = free_map 

        # Scan and list boarder location
        boarder = []
        dir_x = [0, 0, 1, 0, -1] # dx for [stay, down, right, up, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, down, right, up,left]
        for i in range(len(free_map)):
            for j in range(len(free_map[0])):
                if free_map[i][j] == self.team:
                    count = 0
                    for move in range(1,5):
                        nx, ny = i+dir_x[move], j+dir_y[move]
                        if nx < 0 or nx >= len(free_map): continue
                        if ny < 0 or ny >= len(free_map[0]): continue
                        if free_map[nx][ny] == other_team:
                            count += 1
                            break
                    if count:
                        boarder.append((i,j))
        # Group boarder (BFS)
        grouped_boarder = []
        while len(boarder) > 0:
            visited = []
            queue = []
            queue.append(boarder.pop()) 
            while len(queue) > 0:
                n = queue.pop()
                visited.append(tuple(n))
                for move in range(1,5):
                    nx, ny = n[0]+dir_x[move], n[1]+dir_y[move]
                    if (nx,ny) in boarder:
                        boarder.remove((nx,ny))
                        queue.append((nx,ny))
            grouped_boarder.append(visited)

        boarder_centroid = [np.mean(boarder, axis=0) for boarder in grouped_boarder]

        # Assign boarder
        self.assigned = []
        for agent in agent_list:
            x = np.asarray(agent.get_loc())
            dist = [sum(abs(x-centroid)) for centroid in boarder_centroid] # L1 norm
            b = np.argmin(dist)
            self.assigned.append(b)

        # Find path to boarder
        self.route = []
        for idx, agent in enumerate(agent_list):
            target = random.choice(grouped_boarder[self.assigned[idx]])
            route = self.astar.astar_route(agent.get_loc(), target, free_map)
            if route is None:
                self.route.append(None)
            else: 
                self.route.append(route)

        self.grouped_boarder = grouped_boarder
        self.heading_right = [True] * len(agent_list) #: Attr to track directions.
        
    def gen_action(self, agent_list, observation):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        action_out = []
        
        for idx, agent in enumerate(agent_list):
            if not agent.isAlive:
                action_out.append(0)
                continue

            boarder = self.grouped_boarder[self.assigned[idx]]
            route = self.route[idx]
            cur_loc = agent.get_loc()
            if cur_loc in boarder: ## Patrol
                self.route[idx] = None
                a = self.patrol(cur_loc, boarder, self.free_map)
                action_out.append(a)
            elif route is None:
                action_out.append(np.random.randint(5))
            else: ## Navigate to boarder
                step = route.index(cur_loc)
                new_loc = route[step+1]
                if new_loc[1] - cur_loc[1] > 0:   # move right
                    action = 3
                elif new_loc[1] - cur_loc[1] < 0: # move left
                    action = 1
                elif new_loc[0] - cur_loc[0] > 0: # move down
                    action = 2
                elif new_loc[0] - cur_loc[0] < 0: # move up
                    action = 4
                action_out.append(action)
        return action_out

    def patrol(self, loc, boarder, obs):
        x,y = loc
        
        #patrol along the boarder.
        dir_x = [0, 0, 1, 0, -1] # dx for [stay, down, right, up, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, down, right, up,left]
        def cannot_move(nx,ny):
            if nx < 0 or nx >= 20: return True
            elif ny < 0 or ny >= 20: return True
            return obs[nx][ny]==8
        action = [0]
        for a in range(1,5):
            nx = x + dir_x[a]
            ny = y + dir_y[a]
            if cannot_move(nx,ny): continue
            if (nx,ny) in boarder:
                action.append(a)
        return np.random.choice(action)
