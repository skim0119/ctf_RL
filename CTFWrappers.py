import gym
import numpy as np

class FrameStacking(gym.core.Wrapper):

    def __init__(self,env,numFrames=3,lstm=True,**kwargs):
        self.lstm=lstm
        env.reward_range = [-np.inf,np.inf]
        env.metadata = None
        if not lstm:
            self.stackedStates_obs = Stacked_state(numFrames, -1,lstm)
            self.stackedStates_state = Stacked_state(numFrames, -1,lstm)
        else:
            self.stackedStates_obs = Stacked_state(numFrames, 1,lstm)
            self.stackedStates_state = Stacked_state(numFrames, 1,lstm)
        super().__init__(env)

    def reset(self,*args,**kwargs):
        obs = self.env.reset(*args,**kwargs)
        self.stackedStates_obs.initiate(obs)
        self.stackedStates_state.initiate(self.env.get_obs_blue().astype(np.float32))
        return self.stackedStates_obs()

    def step(self,action,*args,**kwargs):
        obs, reward, done, info = self.env.step(action)
        _ = self.stackedStates_state(self.env.get_obs_blue().astype(np.float32))
        return self.stackedStates_obs(obs),reward,done,info

    def GetCentralState(self):
        return self.stackedStates_state()


class Stacked_state:
    def __init__(self, keep_frame, axis,lstm=True):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm=lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame

    def __call__(self, obj=None):
        if obj is None:
            if self.lstm:
                return np.stack(self.stack, axis=self.axis)
            else:
                return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        if self.lstm:
            # print(np.stack(self.stack, axis=self.axis).shape)
            return np.stack(self.stack, axis=self.axis)
        else:
            # print(np.stack(self.stack, axis=self.axis).shape)
            return np.concatenate(self.stack, axis=self.axis)

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
