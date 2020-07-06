import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args


class PPO_DistCritic:
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


class DistCriticCentralKalman:
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

        from network.DistValue_Central_Kalman import V4DistVar
        from network.DistValue_Central_Kalman import train 
        self.model = V4DistVar(input_shape[1:], action_size=5, atoms=atoms)
        self.train = train

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

    def update_network(self, state_input, reward, done, next_state, td_target, b_mean, b_log_var, next_mean, next_log_var):
        inputs = {'state': state_input,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state,
                  'td_target': td_target,
                  'b_mean': b_mean,
                  'b_log_var': b_log_var,
                  'next_mean': next_mean,
                  'next_log_var': next_log_var}
        total_loss, mse, elbo, kloss = self.train(self.model, self.optimizer, inputs)
        return mse, elbo, kloss

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

class DistCriticCentral:
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
        self.model = V4DistVar(input_shape[1:], action_size=5, atoms=4)
        self.train = train

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

    def update_network(self, state_input, reward, done, next_state, td_target):
        inputs = {'state': state_input,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state,
                  'td_target': td_target}
        total_loss, mse, elbo = self.train(self.model, self.optimizer, inputs)
        return mse, elbo

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

class DistCriticCentralCategorical:
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

