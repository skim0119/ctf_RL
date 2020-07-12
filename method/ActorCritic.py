import os

import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args

class PPO_Module:
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
        from network.PPO import V4PPO, train
        self.train = train

        self.model = V4PPO(input_shape[1:], action_size=action_size)

        # Build Network
        #self.model.print_summary()

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
        actor, critics, log_logits = self.model(states)
        actions = tf.random.categorical(log_logits, 1, dtype=tf.int32).numpy().ravel()
        return actions, critics, log_logits

    def update_network(self, state_input, old_log_logit, action, advantage, td_target):
        inputs = {'state': state_input,
                  'old_log_logit': old_log_logit,
                  'action': action,
                  'advantage': advantage,
                  'td_target': td_target}
        total_loss, info = self.train(self.model, self.optimizer, inputs)

        return total_loss, info['actor_loss'], info['critic_loss'], info['entropy']

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

