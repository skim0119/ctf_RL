import numpy as np
import collections
import gym_cap.envs.const as CONST
from gym_cap.envs.const import COLOR_DICT

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
    
    elements = [UNKNOWN, DEAD, TEAM1_BG, TEAM2_BG,
                 TEAM1_GV, TEAM2_GV, TEAM1_UAV, TEAM2_UAV,
                 TEAM1_FL, TEAM2_FL, OBSTACLE]
    map_color = {UNKNOWN: 1, DEAD: 0,
                 TEAM1_BG: 0, TEAM2_BG: 1,
                 TEAM1_GV: 1, TEAM2_GV: -1,
                 TEAM1_UAV: 1, TEAM2_UAV: -1,
                 TEAM1_FL: 1, TEAM2_FL: -1,
                 OBSTACLE: 1}
    
    for element in elements:
        channel = SIX_MAP_CHANNEL[element]
        color = map_color[element]
        image[state[:,:,:,channel]==color] = np.array(COLOR_DICT[element])
        
    return image

def debug():
    """debug
    Include testing code for above methods and classes.
    The execution will start witn __main__, and call this method.
    """

    import gym
    import time
    env = gym.make("cap-v0")
    s = env.reset(map_size=20)

    print('start running')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')

    s = env.reset(map_size=20)

    print('start running v2')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder_v2(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')


if __name__ == '__main__':
    debug()
