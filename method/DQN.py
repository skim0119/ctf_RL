import tensorflow as tf
import tensorflow.keras.layers as layers

import numpy as np
import random

import utility
from utility.utils import store_args

from method.pg import Backpropagation

from method.a3c import a3c
from method.base import put_channels_on_grid
from method.base import put_flat_on_grid
from method.base import put_ctf_state_on_grid
from method.base import initialize_uninitialized_vars as iuv


class Encoder(tf.keras.Model):
    @store_args
    def __init__(self, phi_n=16, action_size=5, trainable=True, name='DQN_encoder'):
        super(Encoder, self).__init__(name=name, trainable=trainable)

        # Feature Encoder
        self.encoder = tf.keras.Sequential([
                layers.SeparableConv2D(
                        filters=16,
                        kernel_size=5,
                        strides=3,
                        padding='valid',
                        depth_multiplier=2,
                        activation='relu',
                    ),
                layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
                layers.Flatten(),
                layers.Dense(units=256, activation='relu'),
            ])

        # Phi Stream
        self.phi_dense = layers.Dense(phi_n, activation='relu')
        self.successor_layer = layers.Dense(action_size, activation='linear')

        # Value Stream
        self.psi_dense = layers.Dense(phi_n, activation='relu'),

        # Advantage Stream
        self.advantage_stream = tf.keras.Sequential([
                layers.Dense(256, activation='relu'),
                layers.Dense(1, activation='linear'),
            ])

    def call(self, inputs):
        latent = self.encoder(inputs)
        phi = self.phi_dense(latent); self.phi = phi
        rewards = self.successor_layer(phi)
        psi = self.psi_dense(latent); self.psi = psi
        values = self.successor_layer(psi)
        advantages = self.advantage_stream(latent)
        with tf.name_scope('qvals'):
            qvals = values + tf.subtract(advantages, tf.reduce_mean(advantages, axis=-1, keepdims=True))

        predict = tf.argmax(qvals, axis=-1)
        
        return qvals, predict, rewards

class DQN:
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        lr=1e-4,
        gamma=0.98,
        entropy_beta=0.01,
        sess=None,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.variable_scope(scope):
                self.states_ = tf.placeholder(shape=input_shape, dtype=tf.float32, name='states')
                self.next_states_ = tf.placeholder(shape=input_shape, dtype=tf.float32, name='next_states')
                self.actions_ = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_hold')
                self.rewards_ = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards_hold')
                self.dones_ = tf.placeholder(shape=[None], dtype=tf.float32, name='dones_hold')

                # Build Network
                model = Encoder(action_size); self.model = model
                self.q, self.predict, self.reward_predict = model(self.states_)
                q_next, _, _ = model(self.next_states_)
                model.summary()
                
                # Build Trainer
                loss = self.build_loss(self.q, self.actions_, self.rewards_, q_next, self.dones_, self.reward_predict)
                optimizer = tf.keras.optimizers.Adam(lr)
                self.gradients = optimizer.get_gradients(loss, model.trainable_variables)
                self.update_ops = optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))

    def build_loss(self, curr_q, actions, rewards, next_q, dones, reward_predict):
        with tf.name_scope('current_Q'):
            oh_action = tf.one_hot(actions, self.action_size, dtype=tf.float32) # [?, num_agent, action_size]
            curr_q = tf.reduce_sum(tf.multiply(curr_q, oh_action), axis=-1) # [?, num_agent]

        with tf.name_scope('reward_pred'):
            reward_predict = tf.reduce_sum(tf.multiply(reward_predict, oh_action), axis=-1)
        
        with tf.name_scope('target_Q'):
            max_next_q = tf.reduce_max(next_q, axis=-1)
            td_target = reward_predict + tf.multiply(self.gamma * max_next_q , (1.0-self.dones_))

        with tf.name_scope('td_error'):
            loss = tf.keras.losses.MSE(td_target, curr_q)
            reward_loss = tf.keras.losses.MSE(rewards, reward_predict)
            softmax_q = tf.nn.softmax(curr_q)
            entropy = -tf.reduce_mean(softmax_q * tf.log(softmax_q))
            total_loss = loss + self.entropy_beta * entropy + 0.5*reward_loss

        self.loss, self.entropy, self.reward_loss = loss, entropy, self.reward_loss
        return total_loss

    def run_network(self, states, return_action=True):
        feed_dict = {self.states_: states}
        action, q = self.sess.run([self.predict, self.q], feed_dict)
        return action, q

    def update_network(self, states, next_states, actions, rewards, dones, global_episodes, writer=None, log=False):
        feed_dict = {
                self.states_: states,
                self.next_states_: next_states,
                self.actions_: actions,
                self.rewards_: rewards,
                self.dones_: dones,
            }
        ops = [self.loss, self.entropy, self.reward_loss, self.update_ops]
        aloss, entropy, rloss, _ = self.sess.run(ops, feed_dict)

        if log:
            # Record losses
            summary = tf.Summary()
            summary.value.add(tag='summary/'+self.scope+'_actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/'+self.scope+'_entropy', simple_value=entropy)
            summary.value.add(tag='summary/'+self.scope+'_reward_loss', simple_value=rloss)

            # Check vanish gradient
            grads = self.sess.run(self.gradients, feed_dict)
            total_counter = 0
            vanish_counter = 0
            for grad in grads:
                total_counter += np.prod(grad.shape) 
                vanish_counter += (np.absolute(grad)<1e-8).sum()
            summary.value.add(tag='summary/grad_vanish_rate', simple_value=vanish_counter/total_counter)
            writer.add_summary(summary,global_episodes)
            writer.flush()

    def initialize_vars(self):
        var_list = self.get_vars
        init = tf.initializers.variables(var_list)
        self.sess.run(init)

    def initiate(self, saver, model_path):
        # Restore if savepoint exist. Initialize everything else
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

    def save(self, saver, model_path, global_step):
        saver.save(self.sess, model_path, global_step=global_step)

    @property
    def get_vars(self):
        return self.model.variables
        

if __name__ == '__main__':
    a = Encoder(5, trainable=False)
    a(tf.placeholder(tf.float32, [None,39,39,12]))
    print(len(a.trainable_weights))
    print(len(a.non_trainable_weights))
    a = Encoder(5, trainable=True)
    a(tf.placeholder(tf.float32, [None,39,39,12]))
    print(len(a.trainable_weights))
    print(len(a.non_trainable_weights))
