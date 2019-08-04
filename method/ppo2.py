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


#from network.attention_ctf import build_network
from network.attention import self_attention
from network.model_V2 import V2_PPO


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
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')

                # Build Network
                model = V2_PPO();  self.model = model
                self.actor, self.logits, self.log_logits, self.critic = model(self.state_input)
                loss = model.build_loss(self.old_logits_, self.action_, self.advantage_, self.td_target_)
                model.feature_network.summary()
                model.summary()

                # Build Trainer
                optimizer = tf.keras.optimizers.Adam(lr)
                self.gradients = optimizer.get_gradients(loss, model.trainable_variables)
                self.update_ops = optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))

            # Summary
            #grad_summary = []
            #for tensor, grad in zip(model.trainable_variables, self.gradients):
            #    grad_summary.append(tf.summary.histogram("%s-grad" % tensor.name, grad))
            #self.merged_grad_summary_op = tf.summary.merge(grad_summary)
            #self.merged_summary_op = self._build_summary(model.trainable_variables)

                self.cnn_summary = self._build_kernel_summary(model.feature_network._layers_snapshot)

    def run_network(self, states):
        feed_dict = {self.state_input: states}
        a_probs, critics, logits = self.sess.run([self.actor, self.critic, self.log_logits], feed_dict)
        actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
        return actions, critics, logits

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
            summary.value.add(tag='summary/actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/critic_loss', simple_value=closs)
            summary.value.add(tag='summary/entropy', simple_value=entropy)

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

    def _build_kernel_summary(self, snapshot):
        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        add_image(snapshot['input'], '1_input', X=6)
        add_image(snapshot['sepCNN1'], '2_sepCNN')
        add_image(snapshot['attention'], '3_attention')
        add_image(snapshot['NLNN'], '4_nonlocal')
        add_image(snapshot['CNN1'], '5_CNN')
        add_image(snapshot['CNN2'], '6_CNN')
        _grid = put_flat_on_grid(snapshot['dense1'][0], 1, 1)
        image_summary.append(tf.summary.image('7_FC1', _grid, max_outputs=1))

        # Collect Summary
        cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        #self.feature_static = snapshot['sepCNN1']
        #self.feature_dynamic = snapshot['attention']
        #self.feature_attention = snapshot['NLNN']
        #labels = tf.one_hot(self.action_, self.action_size, dtype=tf.float32)
        #yc = tf.reduce_sum(logits * labels, axis=1)
        #self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        #self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        #self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]

        return cnn_summary

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
        

