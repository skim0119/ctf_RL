import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args
from utility.logger import *

from network.CVDC_model2 import Central, Decentral
from network.CVDC_model2 import get_gradient, train
from network.CVDC_model2 import loss_central
from network.CVDC_model2 import loss_ppo


class SF_CVDC:
    # Full architecture for centralized value training and
    # decentralized control.
    @store_args
    def __init__(
        self,
        central_obs_shape,
        decentral_obs_shape,
        action_size,
        agent_type,
        save_path,
        atoms=256,
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
        self.model_central = Central(central_obs_shape[1:], atoms=atoms)
        self.save_directory_central = os.path.join(save_path, 'central')
        self.optimizer_central = tf.keras.optimizers.Adam(lr)
        self.checkpoint_central = tf.train.Checkpoint(
                optimizer=self.optimizer_central, model=self.model_central)
        self.manager_central = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central,
                directory=self.save_directory_central,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # Build Network (Decentral)
        for i in range(self.num_agent_type):
            model = Decentral(
                    decentral_obs_shape[1:],
                    action_size=5,
                    atoms=atoms,
                    prebuilt_layers=None if i == 0 else self.dec_models[0])
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

        # PPO Configuration
        self.ppo_config = {
                'eps': tf.constant(0.20, dtype=tf.float32),
                'entropy_beta': tf.constant(0.005, dtype=tf.float32),
                'psi_beta': tf.constant(0.0, dtype=tf.float32),
                'decoder_beta': tf.constant(1e-2, dtype=tf.float32),
                'critic_beta': tf.constant(1, dtype=tf.float32),
                'q_beta': tf.constant(1, dtype=tf.float32),
                'learnability_beta': tf.constant(0.0, dtype=tf.float32),
                }
        # Critic Training Configuration
        self.central_config = {'critic_beta': tf.constant(1.0)}

    def log(self, step, log_weights=True, logs=None):
        # Log specific part of the network
        for i in range(self.num_agent_type):
            model = self.dec_models[i]
            sf_weight = model.sf_v_weight.get_weights()[0]
            sf_q_weight = model.sf_q_weight.get_weights()[0]
            tb_log_histogram(sf_weight, 'monitor/decentral{}_sf_v_weights'.format(i), step)
            tb_log_histogram(sf_q_weight, 'monitor/decentral{}_sf_q_weights'.format(i), step)
            tb_log_histogram(
                    model.beta,
                    'learnability_beta/decentral{}_beta'.format(i),
                    step
                )

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

    def run_network_central(self, states):
        # states: environment state
        # observations: individual (centered) observation
        env_critic, env_feature = self.model_central(states)
        return env_critic, env_feature 

    def run_network_decentral(self, states_list):
        results = []
        for states, model in zip(states_list, self.dec_models):
            num_sample = states.shape[0]
            temp_action = np.zeros(num_sample, dtype=np.int32)
            actor, SF = model([states, temp_action])
            actions = tf.random.categorical(actor["log_softmax"], 1, dtype=tf.int32).numpy().ravel()
            results.append([actions, actor, SF])
        return results

    # Centralize updater
    def update_central(self, datasets, writer=None, log=False, step=None, tag=None):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None
        critic_losses = []
        
        # Get gradients
        model = self.model_central
        optimizer = self.optimizer_central
        for inputs in datasets:
            _, info = train(model, loss_central, optimizer, inputs, self.central_config)
            if log:
                critic_losses.append(info["critic_mse"])
        if log:
            logs = {tag+'central_critic_loss': np.mean(critic_losses)}
            with writer.as_default():
                for name, val in logs.items():
                    tf.summary.scalar(name, val, step=step)
                writer.flush()
    
    # Decentralize updater
    def update_decentral(self, datasets, writer=None, log=False, step=None, tag=None, log_image=False):
        if log:
            assert writer is not None
            assert step is not None
            assert tag is not None
        actor_losses = []
        dec_psi_losses = []
        entropy = []
        decoder_losses = []
        critic_mse = []
        q_losses = []
        reward_mse = []
        multiagent_value_loss = []
        multiagent_reward_loss = []

        learnability_loss = []

        # Get gradients
        for i in range(self.num_agent_type):
            dataset = datasets[i]
            model = self.dec_models[i]
            optimizer = self.dec_optimizers[i]
            for inputs in dataset:
                _, info = train(model, loss_ppo, optimizer, inputs, self.ppo_config)
                #grad, info = get_gradient(model, loss_ppo, inputs, self.ppo_config)
                #optimizer.apply_gradients(zip(grad, model.trainable_variables))
                if log:
                    actor_losses.append(info['actor_loss'])
                    dec_psi_losses.append(info['psi_loss'])
                    entropy.append(info['entropy'])
                    decoder_losses.append(info['generator_loss'])
                    critic_mse.append(info['critic_mse'])
                    q_losses.append(info['q_loss'])
                    reward_mse.append(info['reward_loss'])
                    learnability_loss.append(info['learnability_loss'])
        if log:
            logs = {tag+'dec_actor_loss': np.mean(actor_losses),
                    tag+'dec_psi_loss': np.mean(dec_psi_losses),
                    tag+'dec_entropy': np.mean(entropy),
                    tag+'dec_generator_loss': np.mean(decoder_losses),
                    tag+'dec_critic_loss': np.mean(critic_mse),
                    tag+'dec_q_loss': np.mean(q_losses),
                    tag+'dec_reward_loss': np.mean(reward_mse),
                    tag+'dec_learnability_loss': np.mean(learnability_loss),
                    }
            with writer.as_default():
                self.log(step, logs=logs)
                writer.flush()

        #if log_image:
        if False:
            with writer.as_default():
                for i in range(self.num_agent_type):
                    test_dataset = datasets[i]
                    for inputs in dataset:
                        test_states = inputs['state']
                        break
                    n_state = test_states.shape[0]
                    idx = np.random.randint(low=0, high=n_state, size=8)
                    test_states = tf.gather(test_states, indices=idx, axis=0)
                    _, SF = self.dec_models[i]([test_states, np.zeros(8, dtype=int)])
                    tb_log_ctf_frame(np.stack(test_states, axis=0), tag=f'given states agent{i}', step=step)
                    tb_log_ctf_frame(np.stack(SF['decoded_state'], axis=0), tag=f'decoded states agent{i}', step=step)
                    tb_log_ctf_frame(np.stack(SF['filtered_decoded_state'], axis=0), tag=f'filtered states agent{i}', step=step)
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

