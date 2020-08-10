import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args
from utility.logger import *

from network.DistValue_SF_CVDC import V4SF_CVDC_CENTRAL, V4SF_CVDC_DECENTRAL
from network.DistValue_SF_CVDC import get_gradient, train
from network.DistValue_SF_CVDC import loss_central_critic, loss_reward_central
from network.DistValue_SF_CVDC import loss_ppo, loss_multiagent_critic


class SF_CVDC:
    # Full architecture for centralized value training and
    # decentralized control.
    @store_args
    def __init__(
        self,
        central_obs_shape,
        decentral_obs_shape,
        action_size,
        save_path,
        atoms=4,
        lr=1e-4,
        **kwargs
    ):
        # Set Model
        self.model_central = V4SF_CVDC_CENTRAL(central_obs_shape[1:], atoms=atoms)
        self.model_decentral = V4SF_CVDC_DECENTRAL(decentral_obs_shape[1:], action_size=5, atoms=atoms)

        # Build Network
        self.model_central.print_summary()

        # Build Trainer
        self.save_directory_central = os.path.join(save_path, 'central')
        self.optimizer_central = tf.keras.optimizers.Adam(lr, clipnorm=0.5)
        self.checkpoint_central = tf.train.Checkpoint(
                optimizer=self.optimizer_central, model=self.model_central)
        self.manager_central = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central,
                directory=self.save_directory_central,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)
        self.save_directory_decentral = os.path.join(save_path, 'decentral')
        self.optimizer_decentral = tf.keras.optimizers.Adam(lr, clipnorm=0.5)
        self.checkpoint_decentral = tf.train.Checkpoint(
                optimizer=self.optimizer_decentral, model=self.model_decentral)
        self.manager_decentral = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_decentral,
                directory=self.save_directory_decentral,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # PPO Configuration
        self.ppo_config = {
                'eps': tf.constant(0.2),
                'entropy_beta': tf.constant(0.4),
                'psi_beta': tf.constant(0.02),
                'decoder_beta': tf.constant(1e-4),
                'critic_beta': tf.constant(0.5),
                }
        # Critic Training Configuration
        self.central_config = {
                'psi_beta': tf.constant(0.55),
                'beta_kl': tf.constant(1e-2),
                'elbo_beta': tf.constant(1e-4),
                'critic_beta': tf.constant(1.0),
                }

    def log(self, step):
        sf_weight = self.model_decentral.sf_v_weight.get_weights()[0]
        tb_log_histogram(sf_weight, 'weights/decentral_sf_weights', step)

    def run_network_central(self, states):
        # states: environment state
        # observations: individual (centered) observation
        env_critic, env_feature = self.model_central(states)
        return env_critic, env_feature 

    def run_network_decentral(self, observations):
        actor, dec_SF = self.model_decentral(observations)
        return actor, dec_SF

    # Centralize updater
    def update_reward_prediction(self, inputs, *args):
        _, info = train(self.model_central, loss_reward, self.optimizer_central, inputs)
        return info

    def update_central_critic(self, inputs, *args):
        _, info = train(self.model_central, loss_central_critic, self.optimizer_central, inputs, self.central_config)
        return info
    
    # Decentralize updater
    def update_decentral(self, agent_dataset, team_dataset, log):
        actor_losses = []
        dec_psi_losses = []
        entropy = []
        decoder_losses = []
        critic_mse = []
        q_losses = []
        reward_mse = []
        multiagent_value_loss = []
        multiagent_reward_loss = []

        # Get gradients
        grads = []
        for inputs in agent_dataset:
            grad, info = get_gradient(self.model_decentral, loss_ppo, inputs, self.ppo_config)
            self.optimizer_decentral.apply_gradients(zip(grad, self.model_decentral.trainable_variables))
            #grads.append(grad)
            if log:
                actor_losses.append(info['actor_loss'])
                dec_psi_losses.append(info['psi_loss'])
                entropy.append(info['entropy'])
                decoder_losses.append(info['generator_loss'])
                critic_mse.append(info['critic_mse'])
                q_losses.append(info['q_loss'])
                reward_mse.append(info['reward_loss'])
        '''
        for inputs in team_dataset:
            grad, info = get_gradient(self.model_decentral, loss_multiagent_critic, inputs)
            self.optimizer_decentral.apply_gradients(zip(grad, self.model_decentral.trainable_variables))
            #grads.append(grad)
            if log:
                multiagent_value_loss.append(info['ma_critic'])
                multiagent_reward_loss.append(info['reward_loss'])
        '''
        '''
        # Accumulate gradients
        num_grads = len(grads)
        total_grad = grads.pop(0)
        while grads:
            grad = grads.pop(0)
            for i, val in enumerate(grad):
                total_grad[i] += val

        # Update network
        self.optimizer_decentral.apply_gradients(zip(total_grad, self.model_decentral.trainable_variables))
        '''
                
        logs = {'dec_actor_loss': np.mean(actor_losses),
                'dec_psi_loss': np.mean(dec_psi_losses),
                'dec_entropy': np.mean(entropy),
                'dec_generator_loss': np.mean(decoder_losses),
                'dec_critic_loss': np.mean(critic_mse),
                'dec_q_loss': np.mean(q_losses),
                'dec_reward_loss': np.mean(reward_mse),
                #'dec_ma_critic': np.mean(multiagent_value_loss),
                #'dec_ma_reward': np.mean(multiagent_reward_loss),
                }
        return logs

    # Save and Load
    def initiate(self, verbose=1):
        '''
        last_checkpoint = tf.train.latest_checkpoint(self.save_directory_central)
        if last_checkpoint is None:
            return 0
        else:
            status = self.checkpoint_central.restore(last_checkpoint)
            status.assert_existing_objects_matched()
            #status.assert_consumed()
            last_checkpoint = tf.train.latest_checkpoint(self.save_directory_decentral)
            status = self.checkpoint_decentral.restore(last_checkpoint)
            status.assert_existing_objects_matched() 
            #status.assert_consumed()
            return int(last_checkpoint.split('/')[-1].split('-')[-1])
        '''

        cent_path = self.manager_central.restore_or_initialize()
        decent_path = self.manager_decentral.restore_or_initialize()
        if verbose:
            print('Centralized initialization: {}'.format(cent_path))
            print('Decentralized initialization: {}'.format(decent_path))
        if cent_path == None:
            return 0
        else:
            return int(cent_path.split('/')[-1].split('-')[-1])

    def restore(self):
        status = self.checkpoint_central.restore(self.manager_central.latest_checkpoint)
        status = self.checkpoint_decentral.restore(self.manager_decentral.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager_central.save(checkpoint_number)
        self.manager_decentral.save(checkpoint_number)

