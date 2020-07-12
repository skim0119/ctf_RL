import os
import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np

from utility.utils import store_args

class TDCentral:
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
        # Central encoding
        from network.TD import V4TD
        from network.TD import train 
        self.model = V4TD(input_shape[1:], action_size=5)
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
        inputs = [states]
        v = self.model(inputs)
        return v

    def update_network(self, state_input, reward, done, next_state, td_target):
        inputs = {'state': state_input,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state,
                  'td_target': td_target}
        total_loss = self.train(self.model, self.optimizer, inputs)
        return total_loss

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

