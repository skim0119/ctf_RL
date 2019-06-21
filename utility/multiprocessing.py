#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
import random
from multiprocessing import Process, Pipe

from utility.dataModule import one_hot_encoder as one_hot_encoder

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper, continuous=False, keep_frame=1):
    # If continous == True, automatically reset once the game is over.
    def unstack_frame(frames):
        s = np.concatenate(frames, axis=3)
        return s
    def append_frame(l:list, obj):
        l.append(obj)
        l.pop(0)
        assert len(l) == keep_frame

    parent_remote.close()
    env = env_fn_wrapper.x()

    ctrl_red = env.CONTROL_ALL # If CONTROL_ALL, concat red's view
    stacked_frame = []
    pause = False

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if pause:
                remote.send((ob, reward, done, info))
            else:
                ob, reward, done, info = env.step(data)
                if done:
                    pause = True
                    if continuous:
                        ob = env.reset()
                        pause = False
                ob = one_hot_encoder(ob, env.get_team_blue)
                if ctrl_red:
                    rob = one_hot_encoder(env.get_obs_red, env.get_team_red)
                    ob = np.concatenate([ob, rob], axis=0)
                append_frame(stacked_frame, ob)
                ob = unstack_frame(stacked_frame)
                remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            pause = False
            env.reset(**data)
            ob = one_hot_encoder(env.get_obs_blue, env.get_team_blue)
            if ctrl_red:
                rob = one_hot_encoder(env.get_obs_red, env.get_team_red)
                initial_map = np.concatenate([ob, rob], axis=0)
            else:
                initial_map = ob
            stacked_frame = [np.copy(initial_map) for _ in range(keep_frame)]
            ob = unstack_frame(stacked_frame)
            remote.send(ob)
        elif cmd == 'get_team_blue':
            remote.send(env.get_team_blue)
        elif cmd == 'get_team_red':
            remote.send(env.get_team_red)
        elif cmd == 'blue_win':
            remote.send(env.blue_win)
        elif cmd == 'blue_flag':
            remote.send(env.blue_flag)
        elif cmd == 'red_flag':
            remote.send(env.red_flag)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_full_state':
            remote.send(env.get_full_state)
        else:
            raise NotImplementedError(f'command {cmd} is not found')

class SubprocVecEnv:
    """
    Asynchronous Environment Vectorized run
    """
    def __init__(self, env_fns, keep_frame=1):
        """
        envs: list of gym environments to run in subprocesses
        """

        # Assertions:
        self.waiting = False
        self.closed = False

        nenvs = len(env_fns)
        self.nenvs = nenvs

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            self.ps.append(Process(target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn), False, keep_frame) ) )

        for p in self.ps:
            p.daemon = True # in case of crasehs, process end
            p.start()

        # After process is done
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.num_envs = len(env_fns)

    def step(self, actions=None):
        if actions is None: actions = [None]*self.nenvs
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.concatenate(obs, axis=0), np.stack(rews), np.stack(dones), infos

    def reset(self, **kwargs):
        for remote in self.remotes:
            remote.send(('reset', kwargs))
        return np.concatenate([remote.recv() for remote in self.remotes], axis=0)
    
    def get_full_state(self):
        for remote in self.remotes:
            remote.send(('get_full_state', None))
        return np.stack([remote.recv() for remote in self.remotes]).tolist()

    def get_team_blue(self):
        for remote in self.remotes:
            remote.send(('get_team_blue', None))
        return np.stack([remote.recv() for remote in self.remotes])
        return np.concatenate([remote.recv() for remote in self.remotes], axis=None).tolist()

    def get_team_red(self):
        for remote in self.remotes:
            remote.send(('get_team_red', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def blue_win(self):
        for remote in self.remotes:
            remote.send(('blue_win', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def blue_flag(self):
        for remote in self.remotes:
            remote.send(('blue_flag', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def red_flag(self):
        for remote in self.remotes:
            remote.send(('red_flag', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        """
        Clean up the environments' resources.
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True
            
    def __len__(self):
        return self.nenvs
