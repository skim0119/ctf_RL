import gym
import numpy as np

class SMACWrapper(gym.core.Wrapper):
    """
    Used to give environments a variable for the number of trajectories
    coming from the environment. Default in most environments is 1.
    """
    def __init__(self,env,**kwargs):
        env.action_space = gym.spaces.Discrete(env.n_actions)
        obs,state =env.reset()
        env.observation_space = convert_observation_to_space(obs[0])
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = {'render.modes': []}
        super().__init__(env)

    def reset(self,*args,**kwargs):
        obs, state = self.env.reset(*args,**kwargs)
        return np.vstack(obs)
    def step(self,action,*args,**kwargs):
        reward, terminated, info = self.env.step(action)
        obs = self.get_obs()
        validActions = self.env.get_avail_actions()

        if terminated:
            done = np.asarray([terminated]*self.env.n_agents)
        else:
            done=[]
            for validAction in validActions:
                if validAction[0] == 1:
                    done.append( True)
                else:
                    done.append(False)
            done = np.asarray(done)

        return np.vstack(obs),reward,done,info

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.env.get_avail_agent_actions(agent_id))

        return np.vstack(avail_actions)


    # @property
    # def action_space(self):
    #     return self.n_actions


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


class ActionFiltering(gym.core.Wrapper):
    def __init__(self,env, **kwargs):
        super().__init__(env)

    def reset(self, **kwargs):
        self.tracking_r = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)

        return {"observation":observation,"validActions":self.env.get_avail_actions()}, reward, done, info

class ActionFiltering(gym.core.Wrapper):
    def __init__(self,env, **kwargs):
        super().__init__(env)

    def reset(self, **kwargs):
        self.tracking_r = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action=action)

        return {"observation":observation,"validActions":self.env.get_avail_actions()}, reward, done, info
