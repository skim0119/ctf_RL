import numpy as np
import collections
from gym_cap.envs.const import *

def centering(obs, agents, vision_range, padder=[1,0,0,1,0,0]):
    assert obs.shape[-1] == len(padder)

    length = vision_range*2+1
    states = np.zeros([len(agents), length, length, len(padder)])
    for ch, pad in enumerate(padder):
        states[:,:,:,ch] = pad
    for idx, agent in enumerate(agents):
        x, y = agent.get_loc()
        states[idx, vision_range-x:length-x, vision_range-y:length-y, :] = obs

    return states


class Stacked_state:
    def __init__(self, keep_frame, axis):
        self.keep_frame = keep_frame
        self.axis = axis
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame
    
    def __call__(self, obj=None):
        if obj is None:
            return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        return np.concatenate(self.stack, axis=self.axis)


def oh_to_rgb(state):
    # input: [num_agent, width, height, channel]
    n, w, h, ch = state.shape
    image = np.full(shape=[n, w, h, 3], fill_value=0, dtype=int)
    
    elements = [UNKNOWN, DEAD, TEAM1_BACKGROUND, TEAM2_BACKGROUND,
                 TEAM1_UGV, TEAM2_UGV, TEAM1_UAV, TEAM2_UAV,
                 TEAM1_FLAG, TEAM2_FLAG, OBSTACLE]
    
    for element in elements:
        channel = CHANNEL[element]
        color = REPRESENT[element]
        image[state[:,:,:,channel]==color] = np.array(COLOR_DICT[element])
        
    return image
