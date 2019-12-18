"""Simple agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np
from collections import defaultdict
from utility.RL_Wrapper import TrainedNetwork

from policy.policy import Policy

class UAV(Policy):
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

    def unstack_frame(frames):
        s = np.concatenate(frames, axis=3)
        return s

    def append_frame(l:list, obj):
        l.append(obj)
        l.pop(0)
        assert len(l) == keep_frame

    def __init__(self):
        self.keep_frame = 4
        self.vision_range = 19
        self.network = TrainedNetwork(
                model_name='model/UAV_TRAIN',
                input_tensor='ground/state:0',
                output_tensor='ground/PPO/activation/Softmax:0'
            )

        self.sess = self.network.sess
        self.ground_input = self.network._get_node('ground/state:0')
        self.ground_output = self.network._get_node('ground/PPO/activation/Softmax:0')
        self.uav_input = self.network._get_node('uav/state:0')
        self.uav_output = self.network._get_node('uav/PPO/activation_1/Softmax:0')

    def reload_network(self, path, step=None):
        self.network.reset_network_weight(path=path, step=step)

    def initiate(self, free_map, agent_list):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.agent_list = agent_list
        self.initial_move = True
        self.stacked_frame = None

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

        inputx, inputy = 39, 39
        obs = self.center_pad(observation, width=self.vision_range)

        if self.initial_move:
            self.stacked_frame = [obs for _ in range(self.keep_frame)]
            self.initial_move = False
        else:
            self.stacked_frame.pop(0)
            self.stacked_frame.append(obs)
        obs = np.concatenate(self.stacked_frame, axis=2)

        agent_state = []
        for idx, agent in enumerate(agent_list):
            x, y = agent.get_loc()
            agent_state.append(obs[x:x+inputx, y:y+inputy, :])
        state = np.stack(agent_state)
        uav_state, ground_state = np.split(agent_state, [2], axis=0)
        
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.uav_input: uav_state, self.ground_input: ground_state}
            uav_prob, ground_prob = self.sess.run([self.uav_output, self.ground_output], feed_dict)

        uav_out = [np.random.choice(5, p=prob / sum(prob)) for prob in uav_prob]
        ground_out = [np.random.choice(5, p=prob / sum(prob)) for prob in ground_prob]
        action_out = uav_out + ground_out

        return action_out

    def center_pad(self, m, width, padder=[0,0,0,1,0,0]):
        lx, ly, nch = m.shape
        pm = np.zeros((lx+(2*width),ly+(2*width), nch), dtype=np.int)
        for ch, pad in enumerate(padder):
            pm[:,:,ch] = pad
        pm[width:lx+width, width:ly+width] = m
        return pm

