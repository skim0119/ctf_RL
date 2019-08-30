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
from utility.dataModule import oh_to_rgb

import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy import ndimage

from policy.policy import Policy

class Fix(Policy):
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
                #model_name='model/adapt_train/fixed_subpolicy',
                #input_tensor='main/state:0',
                #output_tensor='main/actor_0/Softmax:0'
                #model_name='model/fix_var1',
                model_name='model/fix_baseline',
                input_tensor='main/state:0',
                output_tensor='main/actor_0/Softmax:0'
            )

        policy0 = self.network._get_node('main/actor_0/Softmax:0')
        policy1 = self.network._get_node('main/actor_1/Softmax:0')
        policy2 = self.network._get_node('main/actor_2/Softmax:0')
        self.ops = [policy0, policy1, policy2]

    def initiate(self, free_map, agent_list):
        self.free_map = free_map
        self.agent_list = agent_list

        self.initial_move = True
        self.stacked_frame = None

    def gen_action(self, agent_list, observation):
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


        # action_out = self.network.get_action(state)
        with self.network.sess.as_default():
            feed_dict = {self.network.state: state}
            prob_0, prob_1, prob_2 = self.network.sess.run(self.ops, feed_dict=feed_dict)
        action_0 = [np.random.choice(5, p=prob/sum(prob)) for prob in prob_0]
        action_1 = [np.random.choice(5, p=prob/sum(prob)) for prob in prob_1]
        action_2 = [np.random.choice(5, p=prob/sum(prob)) for prob in prob_2]

        div01 = 0
        div02 = 0
        div12 = 0
        def div(a, b):
            return -np.mean(a*np.log(b/a))
        prob0 = np.array(list(prob_0.flat))
        prob1 = np.array(list(prob_1.flat))
        prob2 = np.array(list(prob_2.flat))
        div01 = div(prob0, prob1)
        div02 = div(prob0, prob2)
        div12 = div(prob1, prob2)
        #print(div01, div02, div12)
        for p0, p1, p2 in zip(prob_0, prob_1, prob_2):
            n0 = np.argmax(p0)
            n1 = np.argmax(p1)
            n2 = np.argmax(p2)
            if not (n0 == n1 and n1 == n2):
                pass
        #        print('equiv:')
        #        print(p0)
        #        print(p1)
        #        print(p2)
        #        input('')

        return action_0[:2] + action_1[2:3] + action_2[3:]

    def center_pad(self, m, width, padder=[0,0,0,1,0,0]):
        lx, ly, nch = m.shape
        pm = np.zeros((lx+(2*width),ly+(2*width), nch), dtype=np.int)
        for ch, pad in enumerate(padder):
            pm[:,:,ch] = pad
        pm[width:lx+width, width:ly+width] = m
        return pm

