"""Patrolling agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

"""

import numpy as np
import gym_cap.envs.const as const

from policy.policy import Policy

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
    
    def initiate(self, free_map, agent_list):
        self.team = agent_list[0].team
        self.free_map = free_map 
        self.heading_right = [True] * len(agent_list) #: Attr to track directions.
        
    def gen_action(self, agent_list, observation):
        """Action generation method.
        
        This is a required method that generates list of actions corresponding 
        to the list of units. 
        
        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).
            
        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        action_out = []
        
        for agent in agent_list:
            a = self.patrol(agent, self.free_map)
            action_out.append(a)
        
        return action_out

    def patrol(self, agent, obs):
        """Generate 1 action for given agent object."""
        x,y = agent.get_loc()

        #approach the boarder.
        '''if (y > len(self.free_map[0])/2 and 
            self.free_map[x][y-1] == self.free_map[x][y]):
            action = 1
        elif (y < len(self.free_map[0])/2 - 1 and
            self.free_map[x][y+1] == self.free_map[x][y]):
            action = 3
        '''
        
        #patrol along the boarder.
        dir_x = [0, 0, 1, 0, -1] # dx for [stay, down, right, up, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, down, right, up,left]
        enemy = 0 if agent.team else 1
        def cannot_move(x,y,d):
            nx = x + dir_x[d]
            ny = y + dir_y[d]
            if nx < 0 or nx >= 20: return True
            elif ny < 0 or ny >= 20: return True

            return obs[nx][ny]==enemy or obs[nx][ny]==8
        action = [0]
        for a in range(1,5):
            if cannot_move(x,y,a): continue
            action.append(a)
        return np.random.choice(action)

        '''else:
            if (x <= 0 or x >= len(self.free_map)-1):
                self.heading_right[index] = not self.heading_right[index]
                
            #if in the map and have free space at right - go right.
            if (self.heading_right[index] and 
                x+1 < len(self.free_map) and
                obs[x+1][y] == const.TEAM1_BACKGROUND):
                action = 2
            #if in the map and have free space at left - go left.
            elif (not self.heading_right[index] and
                  x > 0 and
                  obs[x - 1][y] == const.TEAM1_BACKGROUND):
                action = 4
            #otherwise - turn around.
            else:
                self.heading_right[index] = not self.heading_right[index]
                
        return action
        '''
