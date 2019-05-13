import numpy as np
import random

from utility.buffer import Replay_buffer

GOAL_BUFFER_SIZE = 5
# goal-replay will augment the experience.
# Adjust number depend on number of episode per train, number of agent, max episode, and buffer size#
GOAL_REPLAY_SAMPLE = 5

class HER:
    """HER
    
    Hindsight Experience Replay
    Based on https://arxiv.org/abs/1707.01495
    Module does not include full algorithm, but an add-on methods to adopt existing A3C code.

    """

    def __init__(self, depth=6, buffer_size=5000):
        """__init__

        :param depth: Number of element stored in each MDP tuple
        :param buffer_size: Capacity of replay buffer
        """
        self.replay_buffer = Replay_buffer(depth=depth, buffer_size=buffer_size)
        self.goal_buffer = Replay_buffer(depth=1, buffer_size=GOAL_BUFFER_SIZE)

    def reward(self, s:tuple, a, g:tuple):
        # -[f_g(s)==0]
        # f can be arbitrary. For simplicity, f is identity
        assert len(s) == len(g)
        assert type(s) == type(g) # tuple
        return int(s==g)
        #return -((s==g)==0)

    def action_replay(self, goal):
        """action_replay

        Push new goal into replay buffer
        The final state of the trajectory is set to new sub-goal

        :param goal:
        """
        self.goal_buffer.append(goal)

    def sample_goal(self, size=GOAL_REPLAY_SAMPLE):
        """sample_goal

        :param size: Number of goal sampled
        """
        if len(self.goal_buffer) <= 0:
            return []
        if len(self.goal_buffer) <= size:
            return self.goal_buffer.buffer[:]
        goal_id = random.sample(range(GOAL_BUFFER_SIZE), size)
        #goal_id = np.random.choice(len(self.goal_buffer), size)
        return [self.goal_buffer[id] for id in goal_id]

    def goal_replay(self, s_traj, a_traj, g):
        """goal_replay

        Take trajectory, action, and goal, return reward

        :param s_traj:
        :param a_traj:
        :param g:
        """
        reward = []
        for s,a in zip(s_traj, a_traj):
            r = self.reward(s,a,g)
            reward.append(r)
            if r == 1:
                break
        if 1 not in reward:
            reward[-1] = -1
        return reward, len(reward)

    def store_transition(self, trajectory:list):
        """store_transition

        :param trajectory:
        :type trajectory: list
        """
        self.replay_buffer.append(trajectory)

    def sample_minibatch(self, size, shuffle=False):
        """sample_minibatch

        :param size:
        :param shuffle:
        """
        if size < len(self.replay_buffer):
            return self.replay_buffer.flush()
        else:
            return self.replay_buffer.pop(size, shuffle)

    def buffer_empty(self):
        """buffer_empty"""
        return self.replay_buffer.empty()

