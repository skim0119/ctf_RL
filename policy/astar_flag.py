"""
    using a-star algorithm to find trajectory and capture flag

    ** The method only works under full observation setting
"""

import numpy as np
import gym_cap.envs.const as const

from policy.policy import Policy

class AStar(Policy):
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
    """

    def __init__(self):
        """Constuctor for policy class.
        
        This class can be used as a template for policy generator.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """

        super().__init__()

    def initiate(self, free_map, agent_list):
        super().initiate(free_map, agent_list)
        self.found_route = []
        self.agent_route = []
        flag_id = const.TEAM2_FLAG if agent_list[0].team==const.TEAM1_BACKGROUND else const.TEAM1_FLAG
        flag = tuple(np.argwhere(free_map==flag_id)[0])
        self.agent_steps = [0]*len(agent_list)
        for idx, agent in enumerate(agent_list):
            start = agent.get_loc()
            self.agent_route.append(self.astar_route(start, flag, free_map))
            self.found_route.append(self.agent_route[idx] is not None)


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
            if not agent.isAlive or not self.found_route[idx]:
                action_out.append(0)
                continue


            cur_loc = agent.get_loc()
            if self.agent_route[idx][self.agent_steps[idx]] != cur_loc:
                self.agent_steps[idx] += 1
            cur_step = self.agent_steps[idx]
            if cur_step >= len(self.agent_route[idx]):
                action_out.append(0)
                continue
            new_loc = self.agent_route[idx][cur_step+1]

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

    def construct_path(self, cameFrom, current):
        """
        return route from start to goal
        """

        total_path = [current]
        while current in cameFrom:
            current = cameFrom[current]
            total_path.append(current)
        total_path.reverse()

        return total_path

    def hScore(self, start, goal):
        """
        Manhattan distance

        """

        dx_1, dy_1 = goal
        dx_2, dy_2 = start
        distance = abs(dx_1 - dx_2) + abs(dy_1 - dy_2)

        return distance

    def hCost(self, neighbour):
        """
        cost of moving to neighbour from current
        """
        return 0

    def astar_route(self, start, goal, board):
        """
        finds route from start to goal
        """

        openSet = set([start])
        closedSet = set()
        cameFrom = {}
        fScore = {}
        gScore = {}
        if len(goal) == 0:
            raise IndexError("Flag is not found for AStar")
        fScore[start] = self.hScore(start, goal)
        gScore[start] = 0

        while openSet:
            min_score = min([fScore[c] for c in openSet])
            for position in openSet:
                if fScore.get(position,np.inf) == min_score:
                    current = position
                    break

            if current == goal:
                return self.construct_path(cameFrom, current)

            openSet.remove(current)
            closedSet.add(current)

            directions = [(1,0),(-1,0),(0,1),(0,-1)] # directions for [right, left, down, up]
            neighbours = []
            for dx, dy in directions:
                x2, y2 = current
                x = x2 + dx
                y = y2 + dy
                if (x >= 0 and x < board.shape[0]) and (y >= 0 and y < board.shape[1]) and board[x,y] != 8:
                    neighbours.append((x, y))

            for neighbour in neighbours:
                if neighbour in closedSet:
                    continue
                tentative_gScore = gScore[current] + self.hCost(neighbour)
                if neighbour not in openSet:
                    openSet.add(neighbour)
                elif tentative_gScore >= gScore[neighbour]:
                    continue
                cameFrom[neighbour] = current
                gScore[neighbour] = tentative_gScore
                fScore[neighbour] = gScore[neighbour] + self.hScore(neighbour, goal)
        return None
