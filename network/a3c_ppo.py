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

from network.base import Deep_layer
from network.pg import Backpropagation
from network.ppo import Loss

from network.a3c import a3c
from network.base import put_channels_on_grid
from network.attention import non_local_nn_2d


class PPO(a3c):
    """PPO

    """
    @store_args
    def __init__(
        self,
        in_size,
        action_size,
        scope,
        lr_actor=1e-4,
        lr_critic=1e-4,
        entropy_beta=0.01,
        sess=None,
        global_network=None,
        tau=None,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."
        if global_network is None: # For primary graph, pipe to self
            self.global_network = self

        with self.sess.as_default(), self.sess.graph.as_default():
            #loss = kwargs.get('loss', Loss.softmax_cross_entropy_selection)
            loss = kwargs.get('loss', Loss.ppo)
            backprop = kwargs.get('back_prop', Backpropagation.asynch_pipeline)

            with tf.variable_scope(scope):
                self._build_placeholder(in_size)
                self.old_logits_ = tf.placeholder(shape=[None], dtype=tf.float32, name='old_logit_holder')

                # get the parameters of actor and critic networks
                self.logits, self.actor, self.critic, self.a_vars, self.c_vars = self._build_network(self.state_input)

                self.kl = self._kl_entropy()

                # Local Network
                train_args = (self.action_, self.advantage_, self.td_target_)
                #loss = loss(self.actor,
                loss = loss(self.actor, self.logits, self.old_logits_,
                        *train_args, self.critic, entropy_beta=entropy_beta)
                self.actor_loss, self.critic_loss, self.entropy = loss

                self.pull_op, self.update_ops, gradients = backprop(
                    self.actor_loss, self.critic_loss,
                    self.a_vars, self.c_vars,
                    self.global_network.a_vars, self.global_network.c_vars,
                    lr_actor, lr_critic,
                    tau,
                    return_gradient=True
                )

            # Summary
            grad_summary = []
            for tensor, grad in zip(self.a_vars+self.c_vars, gradients):
                grad_summary.append(tf.summary.histogram("%s-grad" % tensor.name, grad))
            self.merged_grad_summary_op = tf.summary.merge(grad_summary)
            self.merged_summary_op = self._build_summary(self.a_vars + self.c_vars)

    def run_network(self, states):
        """ run_network
        Parameters
        ----------------
        states : [List/np.array]

        Returns
        ----------------
            action : [List]
            critic : [List]
            logits
        """
        feed_dict = {self.state_input: states}
        a_probs, critics, logits = self.sess.run([self.actor, self.critic, self.logits], feed_dict)
        actions = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs]
        logits = [logit[action] for logit, action in zip(logits, actions)]
        return actions, critics, logits

    def update_global(self, state_input, action, td_target, advantage, old_logit, global_episodes, writer=None, log=False):
        """ update_global

        Run all update and back-propagation sequence given the necessary inputs.

        Parameters
        ----------------
        log : [bool]
             logging option

        """
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}
        self.sess.run(self.update_ops, feed_dict)

        ops = [self.actor_loss, self.critic_loss, self.entropy]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        if log:
            log_ops = [self.cnn_summary,
                       self.merged_grad_summary_op,
                       self.merged_summary_op]
            summaries = self.sess.run(log_ops, feed_dict)
            for summary in summaries:
                writer.add_summary(summary, global_episodes)
            summary = tf.Summary()
            summary.value.add(tag='summary/actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/critic_loss', simple_value=closs)
            summary.value.add(tag='summary/entropy', simple_value=entropy)
            writer.add_summary(summary,global_episodes)

            writer.flush()


    def _build_network(self, input_hold):
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = [] 
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        with tf.variable_scope('actor'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            net = tf.concat([net_static, net_dynamic], axis=-1)
            add_image(net, 'sep_cnn')
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)
            add_image(net, 'attention')

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv1')

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)
            add_image(net, 'conv2')

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            logits = layers.fully_connected(
                net, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        with tf.variable_scope('critic'):
            net = input_hold
            add_image(net, 'input', X=6)

            # Block 1 : Separable CNN
            net_static = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,:3],
                    num_outputs=24,
                    kernel_size=3,
                    depth_multiplier=8,
                )
            net_dynamic = tf.contrib.layers.separable_conv2d(
                    inputs=net[:,:,:,3:],
                    num_outputs=8,
                    kernel_size=3,
                    depth_multiplier=1,
                )
            net = tf.concat([net_static, net_dynamic], axis=-1)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 2 : Self Attention (with residual connection)
            net = non_local_nn_2d(net, 16, pool=False, name='non_local', summary_adder=add_image)

            # Block 3 : Convolution
            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=3)
            net = tf.contrib.layers.max_pool2d(net, 2)

            net = tf.contrib.layers.convolution(inputs=net, num_outputs=64, kernel_size=2)
            net = tf.contrib.layers.max_pool2d(net, 2)

            # Block 4 : Softmax Classifier
            net = tf.layers.flatten(net) 

            critic = layers.fully_connected(
                net, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        self.cnn_summary = tf.summary.merge(image_summary)
        
        # Visualization
        labels = tf.one_hot(self.action_, 5, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
            
        return logits, actor, critic, a_vars, c_vars
