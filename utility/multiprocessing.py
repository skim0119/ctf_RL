#This code is from openai baseline
#https://github.com/openai/baselines/tree/master/baselines/common/vec_env

import numpy as np
from multiprocessing import Process, Pipe

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

def worker(remote, parent_remote, env_fn_wrapper, continuous=False):
    # If continous == True, automatically reset once the game is over.
    parent_remote.close()
    env = env_fn_wrapper.x()
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
                remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            pause = False
            remote.send(ob)
        elif cmd == 'get_team_blue':
            remote.send(env.get_team_blue)
        elif cmd == 'blue_win':
            remote.send(env.blue_win)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError(f'command {cmd} is not found')

class SubprocVecEnv:
    """
    Asynchronous Environment Vectorized run
    """
    def __init__(self, env_fns):
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
                args=(work_remote, remote, CloudpickleWrapper(env_fn))) )

        for p in self.ps:
            p.daemon = True # in case of crasehs, process end
            p.start()

        # After process is done
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self.num_envs = len(env_fns)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_team_blue(self):
        for remote in self.remotes:
            remote.send(('get_team_blue', None))
        return np.concatenate([remote.recv() for remote in self.remotes], axis=None).tolist()

    def blue_win(self):
        for remote in self.remotes:
            remote.send(('blue_win', None))
        return np.stack([remote.recv() for remote in self.remotes]).tolist()

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