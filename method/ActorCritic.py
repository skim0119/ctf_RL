import os

import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args

from network.PPO import V4PPO, train, get_gradient, loss

class PPO_Module:
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
        self.models, self.optimizers, self.checkpoints, self.managers = [], [], [], []
        for i in range(self.num_agent_type):
            # Model defnition
            model = V4PPO(input_shape[1:], action_size=action_size)
            optimizer = tf.keras.optimizers.Adam(lr)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
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
            actor, critics, log_logits = model(states)
            actions = tf.random.categorical(log_logits, 1, dtype=tf.int32).numpy().ravel()
            results.append([actions, critics, log_logits])
        return results

    def update_network(self, train_datasets, log=False, writer=None, step=None, tag=None):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None

        for i in range(self.num_agent_type):
            dataset = train_datasets[i]
            model = self.models[i]
            optimizer = self.optimizers[i]
            actor_losses, critic_losses, entropies = [], [], []
            for inputs in dataset:
                _, info = train(model, optimizer, inputs)
                if log:
                    actor_losses.append(info['actor_loss'])
                    critic_losses.append(info['critic_loss'])
                    entropies.append(info['entropy'])

            if log:
                logs = {f'actor_loss/agent{i}': np.mean(actor_losses),
                        f'critic_loss/agent{i}': np.mean(critic_losses),
                        f'entropy/agent{i}': np.mean(entropies)}
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
        

