
import gym
from gym import spaces
import numpy as np


class RandomPreyActions(gym.core.Wrapper):
    """
    Wrapper wrap observation into dictionary
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space[0]
        self.state_space = self.state_space
        self.action_space_prey = self.action_space[-1]
        self.action_space = self.action_space[0]
        self.n_agents=self.env.n-1


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation.pop(-1)
        return observation

    def step(self, action):
        actionPrey = self.action_space_prey.sample()
        action_list = list(action)
        action_list.append(actionPrey)
        observation, reward, done, info = self.env.step(action=action_list)
        observation.pop(-1)
        reward.pop(-1)
        if isinstance(done,list):
            done.pop(-1)

        return observation, reward, done, info


class MAPFrameStacking(gym.core.Wrapper):
    """
    Wrapper wrap observation into dictionary
    """

    def __init__(self, env,numFrames=3):
        super().__init__(env)
        self.numFrames = numFrames
        self.stackedStates = [Stacked_state(numFrames, 0,True) for i in range(self.n_agents)]
        self.stackedGlobalState = Stacked_state(numFrames, 0,True)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for i,observation_i in enumerate(observation):
            self.stackedStates[i].initiate(observation_i)

        self.stackedGlobalState.initiate(self.env.get_state())
        return [self.stackedStates[i]() for i in range(self.n_agents)]

    def get_state(self):
        return np.expand_dims(self.stackedGlobalState(),axis=0)

    def get_obs(self):
        return [self.stackedStates[i]() for i in range(self.n_agents)]
    def get_avail_agent_actions(self,agent_i):
        return np.ones(self.action_space.n)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)

        return [self.stackedStates[i](observation[i]) for i,obs_i in enumerate(observation)], reward, done, info, np.expand_dims(self.stackedGlobalState(self.env.get_state()),axis=0)


class PredatorPreyTerminator(gym.core.Wrapper):
    """
    Processes the tracked data of the environment.
    In this case it sums the reward over the entire episode.
    """
    def __init__(self,env, **kwargs):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info,state = self.env.step(action=action)
        if reward[0]>0:
            done=[True]*self.n_agents
        return observation, reward[0], np.asarray(done), info,state


class Stacked_state:
    def __init__(self, keep_frame, axis,lstm=False):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm=lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame

    def __call__(self, obj=None):
        if obj is not None:
            self.stack.append(obj)
            self.stack.pop(0)
        if self.lstm:
            # print(np.stack(self.stack, axis=self.axis).shape)
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)
