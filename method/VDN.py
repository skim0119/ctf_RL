import os

import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args

from network.VDN import VDNNet
from network.VDN import train_critic


class VDN_Module:
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        agent_type,
        save_path,
        lr=1e-4,
        **kwargs
    ):
        assert type(agent_type) is list, "Wrong agent type. (e.g. 2 ground 1 air : [2,1])"

        self.num_agent_type = len(agent_type)
        
        # Build model
        self.central_optimizer = tf.keras.optimizers.Adam(lr)
        self.models, self.checkpoints, self.managers = [], [], []
        self.target_models = []
        for i in range(self.num_agent_type):
            # Model defnition
            model = V4PPO(input_shape[1:], action_size=action_size)
            checkpoint = tf.train.Checkpoint(optimizer=self.central_optimizer, model=model)
            manager = tf.train.CheckpointManager(
                    checkpoint=checkpoint,
                    directory=os.path.join(save_path, f'agent_{i}'),
                    max_to_keep=3,
                    keep_checkpoint_every_n_hours=1)
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.checkpoints.append(checkpoint)
            self.managers.append(manager)

    def run_network(self, states_list):
        results = []
        for states, model in zip(states_list, self.models):
            actions, qvals, critic = model(states)
            results.append([actions, critic])
        return results

    def update_network(self, datasets_critic, agent_type_index, log=False, writer=None, step=None, tag=None):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None

        # Train Critic
        critic_losses = []
        for inputs in datasets_critic:
            inputs['agent_type_index'] = agent_type_index
            _, info = train_critic(self.models, self.central_optimizer, inputs)
            if log:
                critic_losses.append(info['critic_loss'])

        if log:
            logs = {'critic_loss': np.mean(critic_losses)}
            with writer.as_default():
                for name, val in logs.items():
                    tf.summary.scalar(tag+name, val, step=step)

    def initiate(self, verbose=1):
        for manager in self.managers:
            path = manager.restore_or_initialize()
            if verbose:
                print('Initialization: {}'.format(path))

        if path == None:
            return 0
        else:
            return int(path.split('/')[-1].split('-')[-1])

    def restore(self):
        for checkpoint, manager in zip(self.checkpoints, self.managers):
            checkpoint.restore(manager.latest_checkpoint)

    def save(self, checkpoint_number):
        for manager in self.managers:
            manager.save(checkpoint_number)
