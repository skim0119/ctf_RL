import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args
from utility.logger import *

from network.counterfactual import V4COMA_d, V4COMA_c
from network.counterfactual import train
from network.counterfactual import loss_central, loss_decentral


class COMA:
    # Full architecture for centralized value training and
    # decentralized control.
    @store_args
    def __init__(
        self,
        state_shape,
        action_size,
        num_agent,
        agent_type,
        save_path,
        atoms=4,
        lr=1e-4,
        **kwargs
    ):
        assert type(agent_type) is list, "Wrong agent type. (e.g. 2 ground 1 air : [2,1])"
        self.num_agent_type = len(agent_type)

        # Set Model
        self.dec_models = []
        self.dec_optimizers = []
        self.dec_checkpoints = []
        self.dec_managers = []

        # Build Network (Central)
        self.model_central = V4COMA_c(state_shape[1:], atoms=atoms, num_agent=num_agent)
        self.save_directory_central = os.path.join(save_path, 'central')
        self.optimizer_central = tf.keras.optimizers.Adam(lr*10)
        self.checkpoint_central = tf.train.Checkpoint(
                optimizer=self.optimizer_central, model=self.model_central)
        self.manager_central = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central,
                directory=self.save_directory_central,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # Build Network (Decentral)
        for i in range(self.num_agent_type):
            model = V4COMA_d(state_shape[1:], action_size=5, atoms=atoms)
            save_directory = os.path.join(save_path, 'decentral{}'.format(i))
            optimizer = tf.keras.optimizers.Adam(lr)
            checkpoint = tf.train.Checkpoint(
                   optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(
                    checkpoint=checkpoint,
                    directory=save_directory,
                    max_to_keep=3,
                    keep_checkpoint_every_n_hours=1)
            self.dec_models.append(model)
            self.dec_optimizers.append(optimizer)
            self.dec_checkpoints.append(checkpoint)
            self.dec_managers.append(manager)

    def log(self, step, log_weights=True, logs=None):
        # Log weights
        if log_weights:
            for var in self.model_central.weights:
                tb_log_histogram(var.numpy(), 'central_weight/'+var.name, step)
            for i in range(self.num_agent_type):
                for var in self.dec_models[i].weights:
                    tb_log_histogram(var.numpy(), 'decentral{}_weight/'.format(i)+var.name, step)

        # Log Information
        # - information must be given in dictionary form
        if logs is not None:
            for name, val in logs.items():
                tf.summary.scalar(name, val, step=step)

    def run_network_central(self, meta_state, meta_action):
        Q, Q_s_a = self.model_central(meta_state, meta_action)
        return Q, Q_s_a

    def run_network_decentral(self, states_list):
        results = []
        for states, model in zip(states_list, self.dec_models):
            actor = model(states)
            results.append(actor)
        return results

    def update(self, dataset_decentral, dataset_central, writer=None, log=False, step=None, tag=None):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None
        actor_losses = []
        critic_losses = []

        # Get gradients
        for i in range(self.num_agent_type):
            dataset = dataset_decentral[i]
            model = self.dec_models[i]
            optimizer = self.dec_optimizers[i]
            for inputs in dataset:
                _, info = train(model, loss_decentral, optimizer, inputs)
                if log:
                    actor_losses.append(info['actor_loss'])
        for inputs in dataset_central:
            _, info = train(self.model_central, loss_central, self.optimizer_central, inputs)
            if log:
                critic_losses.append(info['critic_loss'])
        if log:
            logs = {
                tag+'dec_actor_loss': np.mean(actor_losses),
                tag+'dec_critic_loss': np.mean(critic_losses),
            }
            with writer.as_default():
                self.log(step, logs=logs)
                writer.flush()

    # Save and Load
    def initiate(self, verbose=1):
        cent_path = self.manager_central.restore_or_initialize()
        if verbose:
            print('Centralized initialization: {}'.format(cent_path))

        for i in range(self.num_agent_type):
            path = self.dec_managers[i].restore_or_initialize()
            if verbose:
                print('Decentralized{} initialization: {}'.format(i, path))

        if cent_path == None:
            return 0
        else:
            return int(cent_path.split('/')[-1].split('-')[-1])

    def restore(self):
        status = self.checkpoint_central.restore(self.manager_central.latest_checkpoint)
        for i in range(self.num_agent_type):
            checpoint = self.dec_checkpoints[i]
            manager = self.dec_managers[i]
            status = checkpoint[i].restore(manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager_central.save(checkpoint_number)
        for i in range(self.num_agent_type):
            manager = self.dec_managers[i]
            manager.save(checkpoint_number)

