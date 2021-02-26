import numpy as np

class SMACWrapper:
    def __init__(self, env, numFramesObs=3, numFramesState=1, lstm=False, **kwargs):
        self.env = env
        self.env_info = env.get_env_info()

        self.num_agent = self.env_info['n_agents']
        self.num_action = self.env_info['n_actions']

        self.lstm=lstm
        self.stackedStates_obs = Stacked_state(numFramesObs, 1, lstm)
        #self.stackedStates_states = Stacked_state(numFramesState, 1, False)

    def reset(self):
        self.env.reset()

        # Initiate
        state = self.env.get_state().astype(np.float32)
        obs = np.vstack(self.env.get_obs()).astype(np.float32)

        # Add action one-hot and agent one-hot
        action_id_oh = np.zeros([self.num_agent, self.num_action], dtype=np.float32)
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)

        return self.stackedStates_obs.initiate(obs), state#self.stackedStates_states.initiate(state)

    def step(self,action,*args,**kwargs):
        reward, terminated, info = self.env.step(action)
        validActions = self.get_avail_actions()

        # Find 'done' forr each agent
        if terminated:
            done = np.asarray([terminated]*self.env_info['n_agents'])
        else:
            done=[]
            for validAction in validActions:
                if validAction[0] == 1 and np.all(validAction[1:] == 0):
                    done.append(True)
                else:
                    done.append(False)
            done = np.asarray(done)

        # Update observation and state
        states = self.env.get_state().astype(np.float32)
        #states = self.stackedStates_states(state)

        obs = np.vstack(self.env.get_obs()).astype(np.float32)
        action_id_oh = np.zeros([self.num_agent, self.num_action], np.float32)
        action_id_oh[np.arange(self.num_agent), np.asarray(action)] = 1
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)
        obss = self.stackedStates_obs(obs)

        # Update Information
        if "battle_won" not in info:
            info["battle_won"]=False
        info['valid_action'] = validActions
        info['terminated'] = terminated

        return (obss, states), reward, done, info

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.env_info['n_agents']):
            avail_actions.append(self.env.get_avail_agent_actions(agent_id))
        return np.vstack(avail_actions)


class Stacked_state:
    def __init__(self, keep_frame, axis,lstm=False):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm=lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame
        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)

    def __call__(self, obj=None):
        if obj is None:
            if self.lstm:
                return np.stack(self.stack, axis=self.axis)
            else:
                return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)
