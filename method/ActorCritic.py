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
        from network.PPO import V4PPO, train, get_gradient
        self.train = train
        self.get_gradient = get_gradient

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

    def run_network(self, states):
        actor, critics, log_logits = self.model(states)
        actions = tf.random.categorical(log_logits, 1, dtype=tf.int32).numpy().ravel()
        return actions, critics, log_logits

    def update_network(self, train_dataset, log=False):
        actor_losses, critic_losses, entropies = [], [], []
        grads = []
        for inputs in train_dataset:
            grad, info = self.get_gradient(self.model, inputs)
            grads.append(grad)
            if log:
                actor_losses.append(info['actor_loss'])
                critic_losses.append(info['critic_loss'])
                entropies.append(info['entropy'])

        # Accumulate gradients
        num_grads = len(grads)
        total_grad = grads.pop(0)
        while grads:
            grad = grads.pop(0)
            for i, val in enumerate(grad):
                total_grad[i] += val

        # Update network
        self.optimizer.apply_gradients(zip(total_grad, self.model.trainable_variables))

        #total_loss, info = self.train(self.model, self.optimizer, inputs)

        logs = {'actor_loss': np.mean(actor_losses),
                'critic_loss': np.mean(critic_losses),
                'entropy': np.mean(entropies)}
        return logs

    def initiate(self, verbose=1):
        path = self.manager.restore_or_initialize()
        if verbose:
            print('Initialization: {}'.format(path))
        if path == None:
            return 0
        else:
            return int(path.split('/')[-1].split('-')[-1])

    def restore(self):
        status = self.checkpoint.restore(self.manager.latest_checkpoint)

    def save(self, checkpoint_number):
        self.manager.save(checkpoint_number)
        

