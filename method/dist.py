import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args

class SFK:
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
        # Non-categorial distributional feature encoding
        # Central encoding
        from network.DistValue_SFK import V4SFK_CENTRAL, V4SFK_DECENTRAL
        from network.DistValue_SFK import train 
        from network.DistValue_SFK import loss_central_critic, loss_reward_central
        from network.DistValue_SFK import loss_ppo, loss_multiagent_critic
        self.train = train
        self.loss_central_critic = loss_central_critic
        self.loss_reward = loss_reward_central
        self.loss_ppo = loss_ppo
        self.loss_multiagent_critic = loss_multiagent_critic

        # Set Model
        self.model_central = V4SFK_CENTRAL(central_obs_shape[1:], atoms=atoms)
        self.model_decentral = V4SFK_DECENTRAL(decentral_obs_shape[1:], action_size=5)

        # Build Network
        self.model_central.print_summary()

        # Build Trainer
        self.optimizer_central = tf.keras.optimizers.Adam(lr)
        self.checkpoint_central = tf.train.Checkpoint(
                optimizer=self.optimizer_central, model=self.model_central)
        self.manager_central = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_central,
                directory=os.path.join(save_path, 'central'),
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)
        self.optimizer_decentral = tf.keras.optimizers.Adam(lr)
        self.checkpoint_decentral = tf.train.Checkpoint(
                optimizer=self.optimizer_decentral, model=self.model_decentral)
        self.manager_decentral = tf.train.CheckpointManager(
                checkpoint=self.checkpoint_decentral,
                directory=os.path.join(save_path, 'decentral'),
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)

        # Initiate Model
        self.initiate()

    def run_network_central(self, states, bmean, blogvar):
        # states: environment state
        # bmean, blogvar: belief-state
        # observations: individual (centered) observation
        env_critic, env_feature, env_pred_feature = self.model_central([states, bmean, blogvar])
        return env_critic, env_feature, env_pred_feature

    def run_network_decentral(self, observations, beliefs):
        actor, dec_SF = self.model_decentral([observations, beliefs])
        return actor, dec_SF

    # Centralize updater
    def update_reward_prediction(self, state_input, reward, b_mean, b_log_var, *args):
        inputs = {'state': state_input,
                  'reward': reward,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var}
        reward_loss = self.train(self.model_central, self.loss_reward, self.optimizer_central, inputs)
        return reward_loss

    def update_central_critic(self, state_input, b_mean, b_log_var, td_target, next_mean, next_log_var, *args):
        inputs = {'state': state_input,
                  'td_target': td_target,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var,
                  'next_mean': next_mean,
                  'next_log_var': next_log_var}
        _, info = self.train(self.model_central, self.loss_central_critic, self.optimizer_central, inputs)
        return info
    
    # Decentralize updater
    def update_ppo(self, state_input, belief, old_log_logit, action, td_target, advantage):
        inputs = {'state': state_input,
                  'belief': belief,
                  'old_log_logit': old_log_logit,
                  'action': action,
                  'td_target': td_target,
                  'advantage': advantage,}
        _, info = self.train(self.model_decentral, self.loss_ppo, self.optimizer_decentral, inputs)
        return info

    def update_multiagent_critic(self, states_list, belief, value_central, mask):
        #(TODO)
        states_list = np.reshape(states_list, [600,79,79,6])
        belief = np.reshape(belief, [600,8])
        inputs = {'states_list': states_list,
                  'belief': belief,
                  'value_central': value_central,
                  'mask': mask}
        _, info = self.train(self.model_decentral, self.loss_multiagent_critic, self.optimizer_decentral, inputs)
        return info

    # Misc
    def initiate(self, verbose=1):
        cent_path = self.manager_central.restore_or_initialize()
        decent_path = self.manager_decentral.restore_or_initialize()
        if verbose:
            print('Centralized initialization: {}'.format(cent_path))
            print('Decentralized initialization: {}'.format(decent_path))

    def restore(self):
        status = self.checkpoint_central.restore(self.manager_central.latest_checkpoint)
        status = self.checkpoint_decentral.restore(self.manager_decentral.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager_central.save(checkpoint_number)
        self.manager_decentral.save(checkpoint_number)

class DistCriticCentralKalman:
    # Centralized training with distributional feature encoding.
    # Critic training focus with kalman filter
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        save_path,
        atoms=4,
        lr=1e-4,
        **kwargs
    ):
        # Non-categorial distributional feature encoding
        # Central encoding

        from network.DistValue_Central_Kalman import V4SFK
        from network.DistValue_Central_Kalman import train 
        from network.DistValue_Central_Kalman import loss_psi
        from network.DistValue_Central_Kalman import loss_reward, loss_predictor, loss_decoder
        self.model = V4SFK(input_shape[1:], action_size=5, atoms=atoms)
        self.train = train
        self.loss_psi = loss_psi
        self.loss_reward = loss_reward
        self.loss_predictor = loss_predictor
        self.loss_decoder = loss_decoder

        # Build Network
        self.model.print_summary()

        # Build Trainer
        self.optimizer = tf.keras.optimizers.Adam(lr)
    
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
                checkpoint=self.checkpoint,
                directory=os.path.join(save_path, scope),
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)
        self.initiate()

    def run_network(self, states, bmean, bvar):
        inputs = [states, bmean, bvar]
        v, feature, feat_mean, feat_log_var, decoded, phi, r_pred, psi, pred_mean, pred_log_var = self.model(inputs)
        return v, feature, feat_mean, np.exp(feat_log_var), phi, r_pred, psi, pred_mean, pred_log_var

    def update_reward_prediction(self, state_input, reward, b_mean, b_log_var, *args):
        inputs = {'state': state_input,
                  'reward': reward,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var}
        reward_loss = self.train(self.model, self.loss_reward, self.optimizer, inputs)
        return reward_loss

    def update_kalman(self, state_input, b_mean, b_log_var, next_mean, next_log_var, *args):
        inputs = {'state': state_input,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var,
                  'next_mean': next_mean,
                  'next_log_var': next_log_var}
        kalman_loss = self.train(self.model, self.loss_predictor, self.optimizer, inputs)
        return kalman_loss

    def update_decoder(self, state_input, b_mean, b_log_var, *args):
        inputs = {'state': state_input,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var}
        elbo = self.train(self.model, self.loss_decoder, self.optimizer, inputs)
        return elbo

    def update_sf(self, state_input, td_target, b_mean, b_log_var, *args):
        inputs = {'state': state_input,
                  'td_target': td_target,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var}
        psi_mse = self.train(self.model, self.loss_psi, self.optimizer, inputs)
        return psi_mse

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

class DistCriticCentral:
    # Training centralized value function without kalman filter
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        save_path,
        lr=1e-4,
        **kwargs
    ):
        # Non-categorial distributional feature encoding
        # Central encoding

        from network.DistValue_Central import V4DistVar
        from network.DistValue_Central import train 
        from network.DistValue_Central import loss_psi, loss_reward, loss_decoder
        self.model = V4DistVar(input_shape[1:], action_size=5, atoms=4)
        self.train = train
        self.loss_psi = loss_psi
        self.loss_reward = loss_reward
        self.loss_decoder = loss_decoder

        # Build Network
        self.model.print_summary()

        # Build Trainer
        self.optimizer = tf.keras.optimizers.Adam(lr)
    
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
                checkpoint=self.checkpoint,
                directory=os.path.join(save_path, scope),
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)
        self.initiate()

    def run_network(self, states):
        v, feature, feat_mean, feat_log_var, decoded, phi, r_pred, psi = self.model(states)
        return v, feature, feat_mean, np.exp(feat_log_var), phi, r_pred, psi

    def update_sf(self, state_input, td_target, *args):
        inputs = {'state': state_input,
                  'td_target': td_target}
        psi_loss = self.train(self.model, self.loss_psi, self.optimizer, inputs)
        return psi_loss

    def update_reward_prediction(self, state_input, reward, *args):
        inputs = {'state': state_input,
                  'reward': reward,}
        reward_loss = self.train(self.model, self.loss_reward, self.optimizer, inputs)
        return reward_loss

    def update_decoder(self, state_input, *args):
        inputs = {'state': state_input}
        decoder_loss = self.train(self.model, self.loss_decoder, self.optimizer, inputs)
        return decoder_loss

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

class DistCriticCentralCategorical:
    # C51, Central
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        save_path,
        lr=1e-4,
        **kwargs
    ):
        from network.C51_Central import V2Dist
        from network.C51_Central import train 
        self.model = V2Dist(input_shape[1:], action_size, v_min=-5, v_max=5, atoms=50)
        self.target_model = None #V2Dist(input_shape[1:], action_size, v_min=-5, v_max=5)
        self.train = train

        # Build Network
        self.model.print_summary()
        #self.model.feature_network.summary()

        # Build Trainer
        self.optimizer = tf.keras.optimizers.Adam(lr)
    
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
                checkpoint=self.checkpoint,
                directory=os.path.join(save_path, scope),
                max_to_keep=5,
                keep_checkpoint_every_n_hours=1)
        self.initiate()

    def run_network(self, states):
        v, v_dist = self.model(states)
        return v, v_dist

    def update_network(self, state_input, reward, done, next_state, td_target):
        inputs = {'state': state_input,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state,
                  'td_target': td_target}
        closs = self.train(self.model, self.target_model, self.optimizer,
                           inputs)
        return closs

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)

class DistCriticCategorical:
    # C51, non-central
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        save_path,
        lr=1e-4,
        **kwargs
    ):
        from network.C51 import V2Dist
        from network.C51 import loss as def_loss
        from network.C51 import get_action
        self.model = V2Dist(input_shape[1:], action_size, v_min=-5, v_max=5)
        self.def_loss = def_loss
        self.get_action = get_action

        # Build Network
        self.model.feature_network.summary()

        # Build Trainer
        self.optimizer = tf.keras.optimizers.Adam(lr)
    
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(
                checkpoint=self.checkpoint,
                directory=os.path.join(save_path, scope),
                max_to_keep=3,
                keep_checkpoint_every_n_hours=1)
        self.initiate()

    def run_network(self, states):
        return self.get_action(self.model, states)

    def update_network(
            self, state_input, action, td_target, advantage, old_logit,
            reward, done, next_state, step, writer=None, log=False):
        inputs = {'state': state_input,
                  'action': action,
                  'td_target': td_target,
                  'advantage': advantage,
                  'old_log_logit': old_logit,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state}
        with tf.GradientTape() as tape:
            loss, aloss, closs, entropy = self.def_loss(
                    self.model, **inputs, beta_actor=0.0, beta_critic=1.0, beta_entropy=0.0,
                    return_losses=True, training=True) 
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if log:
            with writer.as_default():
                tf.summary.scalar('summary/'+self.scope+'_actor_loss', aloss, step=step)
                tf.summary.scalar('summary/'+self.scope+'_critic_loss', closs, step=step)
                tf.summary.scalar('summary/'+self.scope+'_entropy', entropy, step=step)
                writer.flush()

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
