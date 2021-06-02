import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args
from utility.logger import *

from network.CVDC_model_PPO import Central, Decentral
from network.CVDC_model_PPO import train
from network.CVDC_model_PPO import loss_central
from network.CVDC_model_PPO import loss_ppo


class PPO_CVDC:
    # Full architecture for centralized value training and
    # decentralized control.
    @store_args
    def __init__(
        self,
        state_shape,
        obs_shape,
        action_space,
        num_agent,
        num_action,
        save_path,
        atoms=256,
        param=None,
        **kwargs
    ):
        assert param is not None

        lr = param['decentral learning rate']
        clr = param['central learning rate']

        # Set Model
        self.num_agent_type = 1 #(TODO) heterogeneous agent
        self.dec_models = []
        self.dec_optimizers = []
        self.dec_checkpoints = []
        self.dec_managers = []

        # Build Network (Central)
        self.model_central = Central([num_agent], state_shape, atoms=atoms, critic_encoder=param['central critic encoder'])
        self.save_directory_central = os.path.join(save_path, 'central')
        if param['central optimizer'] == 'RMSprop':
            self.optimizer_central = tf.keras.optimizers.RMSprop(clr, rho=0.99, epsilon=1e-5)
        elif param['central optimizer'] == 'Adam':
            self.optimizer_central = tf.keras.optimizers.Adam(clr)
        self.checkpoint_central = tf.train.Checkpoint(
                optimizer=self.optimizer_central, model=self.model_central)
        self.manager_central = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central,
                directory=self.save_directory_central,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # Build Network (Central Target)
        self.model_central_target = Central([num_agent], state_shape, atoms=atoms, critic_encoder=param['central critic encoder'])
        self.save_directory_central_target = os.path.join(save_path, 'central_target')
        self.checkpoint_central_target = tf.train.Checkpoint(model=self.model_central_target)
        self.manager_central_target = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central_target,
                directory=self.save_directory_central_target,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # Build Network (Decentral)
        for i in range(self.num_agent_type): 
            model = Decentral(
                obs_shape,
                action_space=action_space,
                atoms=atoms,
                prebuilt_layers=None,
                critic_encoder=param['decentral critic encoder'],
                pi_encoder=param['decentral pi encoder'],
            )
            save_directory = os.path.join(save_path, 'decentral{}'.format(i))
            if param['decentral optimizer'] == 'RMSprop':
                optimizer = tf.keras.optimizers.RMSprop(lr, rho=0.99, epsilon=1e-5)#Adam(lr)
            elif param['decentral optimizer'] == 'Adam':
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

        # PPO Configuration
        self.target_kl = param['target_kl']
        self.ppo_config = {
                'eps': tf.constant(param['eps'], dtype=tf.float32),
                'entropy_beta': tf.constant(param['entropy beta'], dtype=tf.float32),
                'psi_beta': tf.constant(param['psi beta'], dtype=tf.float32),
                'reward_beta': tf.constant(param['reward beta'], dtype=tf.float32),
                'decoder_beta': tf.constant(param['decoder beta'], dtype=tf.float32),
                'critic_beta': tf.constant(param['critic beta'], dtype=tf.float32),
                'q_beta': tf.constant(param['q beta'], dtype=tf.float32),
                'learnability_beta': tf.constant(param['learnability beta'], dtype=tf.float32),
                }

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

    def run_network_central(self, inputs, states):
        env_critic, env_feature = self.model_central(inputs, states)
        return env_critic, env_feature 

    def run_network_central_target(self, inputs, states):
        env_critic, env_feature = self.model_central_target(inputs, states)
        return env_critic, env_feature 

    def run_network_decentral(self, observations, avail_actions):
        #(TODO) heterogeneous agent
        model = self.dec_models[0]
        num_sample = observations.shape[0]
        temp_action = np.ones([num_sample], dtype=int)
        actor, SF = model([observations, temp_action, avail_actions])
        return actor, SF

    # Centralize updater
    def update_central(self, datasets, writer=None, log=False, step=None, tag=None, param=None):
        critic_losses = []
        
        # Get gradients
        model = self.model_central
        optimizer = self.optimizer_central
        for inputs in datasets:
            _, info = train(model, loss_central, optimizer, inputs, param['central gradient global norm'])
            critic_losses.append(info["critic_mse"])
        print(np.mean(critic_losses))
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None
            logs = {tag+'central_critic_loss': np.mean(critic_losses)}
            with writer.as_default():
                for name, val in logs.items():
                    tf.summary.scalar(name, val, step=step)
                writer.flush()
    
    # Decentralize updater
    def update_decentral(self, datasets, writer=None, log=False, step=None, tag=None, param=None):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None
        actor_losses = []
        entropy = []
        critic_mse = []

        grad_norms = []
        approx_kls = []
        approx_ents = []

        # Get gradients
        for i in range(self.num_agent_type):
            dataset = datasets[i]
            model = self.dec_models[i]
            optimizer = self.dec_optimizers[i]
            for inputs in dataset:
                _, info = train(model, loss_ppo, optimizer, inputs, param['decentral gradient global norm'], self.ppo_config)
                if log:
                    actor_losses.append(info['actor_loss'])
                    entropy.append(info['entropy'])
                    critic_mse.append(info['critic_mse'])
                    grad_norms.append(info["grad_norm"])
                    approx_kls.append(info['approx_kl'])
                    approx_ents.append(info['approx_ent'])
                if info['approx_kl'] > 1.5 * self.target_kl:
                    break
        if log:
            logs = {tag+'dec_actor_loss': np.mean(actor_losses),
                    tag+'dec_entropy': np.mean(entropy),
                    tag+'dec_critic_loss': np.mean(critic_mse),
                    'grad_norm': np.mean(grad_norms),
                    'approx_kl': np.mean(approx_kls),
                    'approx_ent': np.mean(approx_ents),
                    }
            with writer.as_default():
                self.log(step, logs=logs)
                writer.flush()

    def update_target(self):
        self.model_central_target.set_weights(self.model_central.get_weights())

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
        status = self.checkpoint_central_target.restore(self.manager_central_target.latest_checkpoint)
        for i in range(self.num_agent_type):
            checpoint = self.dec_checkpoints[i]
            manager = self.dec_managers[i]
            status = checkpoint[i].restore(manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager_central.save(checkpoint_number)
        self.manager_central_target.save(checkpoint_number)
        for i in range(self.num_agent_type):
            manager = self.dec_managers[i]
            manager.save(checkpoint_number)

