import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import random

import utility


class DQN:
    """Deep Q-Network Implementation for multi-agent usage

    This module contains building network and pipelines to use.

    Attributes:
        @ Private
        _build_Q_network:
        _build_training:

        @ Public
        run_network:
        update_target:
        pull_global:
    Todo:
        pass

    """
    def __init__(self,
                 in_size,
                 action_size,
                 scope,
                 num_agent,
                 trainer=None,
                 tau=0.001,
                 gamma=0.99,
                 grad_clip_norm=0,
                 global_step=None,
                 initial_step=0,
                 sess=None,
                 target_network=None):
                
        """ Initialize AC network and required parameters

        Keyword arguments:
            pass

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:
            * Separate the building trainsequence to separete method.
            * Organize the code with pep8 formating
        """

        # Class Environment
        self.sess = sess
        if target_network is None:
            self.target_network = self
            self.tau = 1.0
        else:
            self.target_network = target_network
            self.tau = tau

        # Parameters & Configs
        self.in_size = in_size
        self.action_size = action_size
        self.scope = scope
        self.trainer = trainer
        self.num_agent = num_agent
        self.grad_clip_norm = grad_clip_norm
        self.global_step = global_step
        self.initial_step = initial_step
        self.gamma = gamma
        
        with tf.variable_scope(scope), tf.device('/gpu:0'):
            self._build_Q_network()
            self.graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            if scope != 'target':
                self._build_training()
                self._build_pipeline()
                            
    def _build_Q_network(self):
        """_build_Q_network
        The network recieves a state of all agencies, and they are represented in [-1,4,19,19,11]
        (number of agencies and channels are subjected to change)

        Series of reshape is require to evaluate the action for each agent.
        """
        in_size = [None, self.num_agent] + self.in_size[1:]
        self.state_input_ = tf.placeholder(shape=in_size,dtype=tf.float32, name='state')
        with tf.name_scope('input_pipeline'):
            n_entry = tf.shape(self.state_input_)[0]
            n_row = tf.shape(self.state_input_)[0] * tf.shape(self.state_input_)[1]
            flat_shape = [n_row] + self.in_size[1:]
            net = tf.reshape(self.state_input_, flat_shape)
        net = layers.conv2d(net , 32, [5,5],
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME')
        net = layers.max_pool2d(net, [2,2])
        net = layers.conv2d(net, 64, [3,3],
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME')
        net = layers.max_pool2d(net, [2,2])
        net = layers.conv2d(net, 64, [2,2],
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer_conv2d(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME')
        # Separate value/advantage stream
        adv_net, value_net = tf.split(net, 2, 3)
        adv_net, value_net = layers.flatten(adv_net), layers.flatten(value_net)
        adv_net = layers.fully_connected(adv_net, self.action_size, activation_fn=None)
        value_net = layers.fully_connected(value_net, 1, activation_fn=None)
        with tf.name_scope('concat'):
            net = value_net + tf.subtract(adv_net, tf.reduce_mean(adv_net, axis=1, keepdims=True))
        with tf.name_scope('rebuild'):
            self.Qout = tf.reshape(net, [-1, self.num_agent, self.action_size])
            self.predict = tf.argmax(self.Qout,2)

    def _build_training(self):
        """_build_training
        Build training sequence for DQN
        Use mask to deprecate dead agency
        """
        self.targetQ_ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_ = tf.placeholder(shape=[None, self.num_agent],dtype=tf.int32)
        self.mask_ = tf.placeholder(shape=[None, self.num_agent], dtype=tf.float32)

        with tf.name_scope('Q'):
            oh_action = tf.one_hot(self.action_, self.action_size, dtype=tf.float32) # [?, num_agent, action_size]
            self.Q_ind = tf.reduce_sum(tf.multiply(self.Qout, oh_action), axis=-1) # [?, num_agent]
            self.Q_sum = tf.reduce_sum(self.Q_ind*self.mask_, axis=-1)
        
        with tf.name_scope('Q_train'):
            self.td_error = tf.square(self.targetQ_-self.Q_sum)
            self.loss = tf.reduce_mean(self.td_error)
            self.entropy = -tf.reduce_sum(tf.nn.softmax(self.Qout) * tf.log(tf.nn.softmax(self.Qout)))
            self.grads = tf.gradients(self.loss, self.graph_vars)

        self.update= self.trainer.apply_gradients(zip(self.grads, self.graph_vars))

    def _build_pipeline(self):
        op_push = [target_var.assign(this_var*self.tau + target_var*(1.0-self.tau)) for target_var, this_var in zip(self.target_network.graph_vars, self.graph_vars)]
        self.op_push = tf.group(op_push)

    def run_network(self, state):
        """run_network
        Choose Action

        :param state:
        """
        return self.sess.run(self.predict, feed_dict={self.state_input_:state}).tolist()

    def update_full(self, states0, actions, rewards, states1, dones, masks):
        n_entry = len(states0)
        q1 = self.sess.run(self.predict, feed_dict={self.state_input_:states1})
        q2 = self.sess.run(self.target_network.Qout, feed_dict={self.target_network.state_input_:states1})
        end_masks = -(dones-1)
        dq = np.zeros_like(q1)
        for idx in range(self.num_agent):
            dq[:,idx] = q2[range(n_entry),idx,q1[:,idx]]
        
        dq = np.sum(dq*masks, axis=1)
        targetQ = rewards + (self.gamma * dq * end_masks)

        feed_dict = {self.state_input_ : states0,
                     self.targetQ_ : targetQ,
                     self.action_ : actions,
                     self.mask_ : masks}
        loss, entropy, _ = self.sess.run([self.loss, self.entropy, self.update], feed_dict)

        self.sess.run(self.op_push)

        return loss, entropy
