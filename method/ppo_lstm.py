"""
a3c.py module includes basic modules and implementation of A3C for CtF environment.

Some of the methods are left unimplemented which makes the a3c module to be parent abstraction.

Script contains example A3C

TODO:
    - Include gradient and weight histograph for nan debug
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from method.pg import Backpropagation

from method.a3c import a3c
from method.base import put_channels_on_grid
from method.base import put_flat_on_grid
from method.base import put_ctf_state_on_grid
from method.base import initialize_uninitialized_vars as iuv

from network.model_lstm import PPO_LSTM_V1


class PPO:
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        lr=1e-4,
        sess=None,
        target_network=None,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name='state')
                self.prev_reward_ = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='prereward_hold')
                self.prev_action_ = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='preaction_hold')
                self.hidden_ = [tf.placeholder(shape=[None, 256], dtype=tf.float32, name='hiddenh_hold'),
                                tf.placeholder(shape=[None, 256], dtype=tf.float32, name='hiddenc_hold')]

                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')

                # Build Network
                model = PPO_LSTM_V1(action_size);  self.model = model
                self.actor, self.logits, self.log_logits, self.critic = model(self.state_input)
                self.hidden = model
                loss = model.build_loss(self.old_logits_, self.action_, self.advantage_, self.td_target_)
                model.feature_network.summary()
                model.summary()

                # Build Trainer
                optimizer = tf.keras.optimizers.Adam(lr)
                self.gradients = optimizer.get_gradients(loss, model.trainable_variables)
                self.update_ops = optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))

    def run_network(self, states, return_action=True):
        feed_dict = {self.state_input: states}
        a_probs, critics, logits = self.sess.run([self.actor, self.critic, self.log_logits], feed_dict)
        if return_action:
            actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
            return actions, critics, logits
        else:
            return a_probs, critics, logits

    def update_network(self, state_input, action, td_target, advantage, old_logit, global_episodes, writer=None, log=False):
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}
        self.sess.run(self.update_ops, feed_dict)

        if log:
            ops = [self.model.actor_loss, self.model.critic_loss, self.model.entropy]
            aloss, closs, entropy = self.sess.run(ops, feed_dict)

            log_ops = [self.cnn_summary]
                       #self.merged_grad_summary_op,
                       #self.merged_summary_op]
            summaries = self.sess.run(log_ops, feed_dict)
            for summary in summaries:
                writer.add_summary(summary, global_episodes)
            summary = tf.Summary()
            summary.value.add(tag='summary/'+self.scope+'_actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/'+self.scope+'_critic_loss', simple_value=closs)
            summary.value.add(tag='summary/'+self.scope+'_entropy', simple_value=entropy)

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
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        

