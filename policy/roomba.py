"""Simple agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np
from collections import defaultdict

from policy.policy import Policy

class Roomba(Policy):
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
        policy: Method to determine the action based on observation for a single unit
        scan : Method that returns the dictionary of object around the agent

    Variables:
        exploration : exploration rate
        previous_move : variable to save previous action
    """

    def initiate(self, free_map, agent_list):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.free_map = free_map

        self.random = np.random
        self.exploration = 0.1
        self.previous_move = self.random.randint(0, 5, len(agent_list)).tolist()

        self.team = agent_list[0].team

        self.flag_location = None
        self.enemy_flag_code = 7
        self.enemy_code = 1
        self.enemy_agent_code = 4
        self.team_code = 0

        self.enemy_range = 5 # Range to see around and avoid enemy
        self.flag_range = 5  # Range to see the flag

    def gen_action(self, agent_list, observation, free_map=None):
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

        # Expand the observation with wall
        padding = max(self.enemy_range, self.flag_range)
        obs = self.center_padding(observation, padding)

        for idx, agent in enumerate(agent_list):
            # Initialize Variables
            x, y = agent.get_loc()
            x += padding
            y += padding
            self_centered = obs[x+1-padding:x+padding,
                                y+1-padding:y+padding] # limited view for the agent
            action = self.policy(agent, self_centered, idx)
            action_out.append(action)

        return action_out

    def policy(self, agent, obs, agent_id):
        """ Policy

        This method generate an action for given agent.
        Agent is given with limited vision of a field.
        This method provides simple protocol of movement based on the agent's location and it's vision.

        Protocol :
            1. Scan the area with flag_range:
                - Flag within radius : Set the movement towards flag
                - No Flag : random movement
            2. Scan the area with enemy_range:
                - Enemy in the direction of movement
                    - If I'm in enemy's territory: reverse direction
                    - Continue
                - Else: contonue moving in the direction
            3. Random exploration
                - 0.1 chance switching direction of movement
                - Follow same direction
                - Change direction if it heats the wall
        """

        # Continue the previous action
        action = self.previous_move[agent_id]
        x, y = obs.shape
        x //= 2; y //= 2

        dir_x = [0, 0, 1, 0, -1] # dx for [stay, up, right, down, left]
        dir_y = [0,-1, 0, 1,  0] # dy for [stay, up, right, down ,left]
        blocking = lambda d: obs[x+dir_x[d]][y+dir_y[d]] in [8, 2]  

        field = self.scan(obs)
        elements = field.keys()
        # 1. Set direction to flag
        if self.enemy_flag_code in elements: # Flag Found
            # move towards the flag
            fx, fy = field[self.enemy_flag_code][0]
            fx -= x; fy -= y
            action_pool = []
            if fy > 0: # move down
                action_pool.append(3)
            if fy < 0: # move up
                action_pool.append(1)
            if fx > 0: # move left
                action_pool.append(2)
            if fx < 0: # move right
                action_pool.append(4)
            if action_pool == []:
                action_pool = [0]
            action = np.random.choice(action_pool)
        
        # 2. Scan with enemy range
        opposite_move = [0, 3, 4, 1, 2]
        for ex, ey in field.get(self.enemy_agent_code, []):
            if self.free_map[agent.get_loc()] != agent.team:
                ex -= x; ey -= y
                if (ey > 0 and abs(ex) < 2 and action == 3) or \
                   (ey < 0 and abs(ex) < 2 and action == 1) or \
                   (ex > 0 and abs(ey) < 2 and action == 2) or \
                   (ex < 0 and abs(ey) < 2 and action == 4):
                    action = opposite_move[action]
            else:
                if ey > 0 and ex == 0: # move down
                    action = 3
                elif ey < 0 and ex == 0: # move up
                    action = 1
                elif ex > 0 and ey == 0: # move left
                    action = 2
                elif ex < 0 and ey == 0: # move right
                    aciton = 4


        # 3. Exploration
        if action == 0 or np.random.random() <= self.exploration: # Exploration
            action = np.random.randint(1,5)
        # Checking obstacle
        if blocking(action): # Wall or other obstacle
            action_pool = [move for move in range(1,5) if not blocking(move)]
            if action_pool == []:
                action_pool = [0]
            action = np.random.choice(action_pool)

        self.previous_move[agent_id] = action

        return action

    def scan(self, view, exclude=[-1, 8]):
        """
        This function returns the dictionary of locations for each element by its type.
            key : field element (int)
            value : list of element's coordinate ( (x,y) tuple )
        """

        objects = defaultdict(list)
        dx, dy = view.shape
        for i in range(dx):
            for j in range(dy):
                if view[i][j] in exclude: 
                    continue
                objects[view[i][j]].append((i,j))

        return objects

    def center_padding(self, m, width, padder=8):
        lx, ly = m.shape
        pm = np.empty((lx+(2*width),ly+(2*width)), dtype=np.int)
        pm[:] = padder
        pm[width:lx+width, width:ly+width] = m
        return pm

