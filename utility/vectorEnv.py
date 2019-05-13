# Vectorize the environment to enable multiprocessing the environment rollout
# Try to increase training efficiency when the training is having a bottleneck in environment rollout
# Code does not include the tensorflow graph: separate algorithm is require to generate the action.
# If not provided, it will use random action.
# Source : https://github.com/openai/baselines/tree/master/baselines/a2c

# This class is to run multiple environments at the same time.

from multiprocessing import Process, Pipe
import numpy as np

def worker(remote, env_wrapper, map_size, policy_red):
    env = env_wrapper
    policy_red = policy_red
    map_size = map_size
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if done:
                remote.send((ob, reward, done))
            else:
                ob, reward, done, _ = env.step(data)
                #if done:
                #    ob = env.reset(map_size=MAP_SIZE, policy_red=policy_red)
                ob = env._env # comment this line to make partial observable
                remote.send((ob, reward, done))
        elif cmd == 'reset':
            done = False
            ob = env.reset(map_size=map_size, policy_red=policy_red.PolicyGen(env.get_map, env.get_team_red))
            ob = env._env
            remote.send((ob, env.get_team_blue))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'won':
            remote.send(env.blue_win)
        elif cmd == 'render':
            pass
        elif cmd == 'renew':
            # renew weight of the policy
            # policy must support reset weight method
            policy_red.reset_network()
        elif cmd == 'change_red_policy':
            # Change policy of red with given data
            policy_red=data
        elif cmd == 'change_mapsize':
            # Change map_size
            map_size=data
        else:
            raise NotImplementedError

class SubprocVecEnv:
    # Subprocess Vector Environment
    # https://github.com/openai/baselines/tree/master/baselines/a2c (source)
    # with some modificatoins
    def __init__(self, nenvs, env_fns, map_size, initial_reds):
        self.nenvs = nenvs
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(self.work_remotes[idx], env_fns[idx], map_size, initial_reds[idx]))
                   for idx in range(nenvs)]
        for pidx, p in enumerate(self.ps):
            p.start()
            print(f"Process {pidx} Initiated")

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, team = zip(*results)
        return np.stack(obs), np.stack(team)
    
    def won(self):
        for remote in self.remotes:
            remote.send(('won', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            
    def change_red_policy(self, policy):
        for remote in self.remotes:
            remote.send(('change_red_policy', policy))
            
    def change_mapsize(self, ms):
        for remote in self.remotes:
            remote.send(('change_mapsize', ms))

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    @property
    def num_envs(self):
        return len(self.remotes)