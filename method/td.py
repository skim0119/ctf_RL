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
        atoms,
        lr=1e-4,
        **kwargs
    ):
        # Central encoding
        from network.TD import V4TD
        from network.TD import train 
        from network.TD import loss, loss_decoder
        self.model = V4TD(input_shape[1:], action_size=5, atoms=atoms)
        self.train = train
        self.loss_td = loss
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
        inputs = [states]
        v = self.model(inputs)
        return v

    def update_network(self, state_input, reward, done, next_state, td_target):
        inputs = {'state': state_input,
                  'reward': reward,
                  'done': done,
                  'next_state': next_state,
                  'td_target': td_target}
        total_loss = self.train(self.model, self.loss_td, self.optimizer, inputs)
        return total_loss

    def update_decoder(self, state_input):
        inputs = {'state': state_input}
        loss = self.train(self.model, self.loss_decoder, self.optimizer, inputs)
        return loss

    def initiate(self):
        return self.manager.restore_or_initialize()

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

