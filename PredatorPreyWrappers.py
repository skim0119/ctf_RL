import gym
from gym import spaces
import numpy as np
from environments.Common import Stacked_state
from utils.utils import MovingAverage

class MAPFrameStacking(gym.core.Wrapper):
    """
    Wrapper wrap observation into dictionary
    """

    def __init__(self, env,numFrames=3):
        super().__init__(env)
        self.numFrames = numFrames
        self.stackedStates = [Stacked_state(numFrames, 0,True) for i in range(self.env.n)]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for i,observation_i in enumerate(observation):
            self.stackedStates[i].initiate(observation_i)

        return [self.stackedStates[i]() for i in range(self.env.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)

        return [self.stackedStates[i](observation[i]) for i,obs_i in enumerate(observation)], reward, done, info


class HomogeneousMultiAgentWrapper(gym.core.Wrapper):
    """
    Wrapper wrap observation into dictionary
    """

    def __init__(self, env,name="state"):
        super().__init__(env)

        self.name = name
        self.observationSpace = spaces.Dict({
            'obs':spaces.Dict({
                self.name: self.observation_space
            }),
            'data':spaces.Dict({
            })
        })
        self.actionSpace = self.action_space
        self.nTrajs = 8

    def reset(self, **kwargs):
        observation,state = self.env.reset(**kwargs)
        validActions = self.env.get_avail_actions()
        obsList = []
        for observation_i,validActions_i in zip(observation,validActions):
            obsList.append({"obs":{self.name:np.expand_dims(observation_i,axis=0),
                            "validActions":validActions_i},"data":{}})
        return obsList

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)
        validActions = self.env.get_avail_actions()
        obsList = []
        for observation_i,validActions_i in zip(observation,validActions):
            obsList.append({"obs":{self.name:np.expand_dims(observation_i,axis=0),
                            "validActions":validActions_i},"data":{}})
        return obsList, reward, done, info

class ArrayDoneMA(gym.core.Wrapper):
    """Method is used to convert to integers if the environment can't handle arrays as actions... """
    def step(self, action):
        obs,r,d,info = self.env.step(action)
        return obs, r, np.asarray([d]*self.nTrajs), info


class ArrayRewardMA(gym.core.Wrapper):
    """Method is used to convert to integers if the environment can't handle arrays as actions... """
    def step(self, action):
        obs,r,d,info = self.env.step(action)
        return obs, np.asarray([r]*self.nTrajs), d, info

class RewardLoggingMAP(gym.core.Wrapper):
    """
    Processes the tracked data of the environment.
    In this case it sums the reward over the entire episode.
    """
    def __init__(self,env,loggingPeriod=100, **kwargs):
        super().__init__(env)
        self.runningReward = MovingAverage(loggingPeriod)
        self.captureAttemptMA = MovingAverage(loggingPeriod)
        self.rewardTracker = []
        self.captureAttemptTracker = []

    def reset(self, **kwargs):
        if len(self.rewardTracker) == 0:
            pass
        else:
            self.runningReward.append(sum(self.rewardTracker))
            self.captureAttemptMA.append(sum(self.captureAttemptTracker))
        self.rewardTracker = []
        self.captureAttemptTracker = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)
        self.rewardTracker.append(reward)
        self.captureAttemptTracker.append(1 if 5 in action else 0)
        return observation, reward, done, info

    def getLogging(self):
        dict = self.env.getLogging()
        localDict = {
            "TotalReward":self.runningReward(),
            "CaptureAttempts":self.captureAttemptMA()}
        dict.update(localDict)
        return dict
