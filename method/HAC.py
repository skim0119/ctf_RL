import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from network.base import Deep_layer 


class HAC:
    """Actor Critic Network Implementation for A3C (Tensorflow)

    This module contains building network and pipelines to sync with global network.
    Global network is expected to have same network structure.
    Actor Critic is implemented with convolution network and fully connected network.
        - LSTM will be added depending on the settings

    Attributes:
        pass
    Todo:
        pass

    """
    @store_args
    def __init__(self,
                 local_state_shape,
                 shared_state_shape,
                 action_size,
                 scope,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 entropy_beta=0.001,
                 critic_beta=1.0,
                 explicit_policy=True,
                 sess=None,
                 global_network=None):
        """ Initialize AC network and required parameters

        Keyword arguments:
            explicit_policy: If false, use single critic network and Q value for each action.

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:
            * Separate the building trainsequence to separete method.
            * Organize the code with pep8 formating

        """

        with tf.variable_scope(scope):
            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input_ = tf.placeholder(shape=local_state_shape, dtype=tf.float32, name='state')
            self.gps_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='gps_state')
            self.goal_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='goal_state')

            # get the parameters of actor and critic networks
            if explicit_policy:
                self.actor, self.critic, self.a_vars, self.c_vars = self._build_policy_network(self.state_input_, self.gps_state_, self.goal_state_)

                if global_network is not None:
                    self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
                    self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

                    self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                    self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                    self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')

                    with tf.name_scope('train'):
                        # Critic (value) Loss
                        td_error = self.td_target_ - self.critic
                        self.entropy = -tf.reduce_mean(self.actor * tf.log(self.actor), name='entropy')
                        self.critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

                        # Actor Loss
                        action_OH = tf.one_hot(self.action_, action_size)
                        obj_func = tf.log(tf.reduce_sum(self.actor * action_OH, 1))
                        exp_v = obj_func * self.advantage_
                        self.actor_loss = tf.reduce_mean(-exp_v, name='actor_loss') - entropy_beta * self.entropy
                        self.total_loss = critic_beta * self.critic_loss + self.actor_loss - entropy_beta * self.entropy

                    with tf.name_scope('local_grad'):
                        a_grads = tf.gradients(self.actor_loss, self.a_vars)
                        c_grads = tf.gradients(self.critic_loss, self.c_vars)

                    # Sync with Global Network
                    with tf.name_scope('sync'):
                        pull_a_vars_op = self._build_pull(self.a_vars, self.global_network.a_vars)
                        pull_c_vars_op = self._build_pull(self.c_vars, self.global_network.c_vars)
                        self.pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

                        update_a_op = self._build_push(a_grads, self.global_network.a_vars, self.actor_optimizer)
                        update_c_op = self._build_push(c_grads, self.global_network.c_vars, self.critic_optimizer)
                        self.update_ops = tf.group(update_a_op, update_c_op)
            else:
                self.q, self.q_vars = self._build_q_network(self.state_input_, self.gps_state_, self.goal_state_)
                self.predict = tf.argmax(self.q, axis=-1)

                if global_network is not None:
                    self.optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')

                    self.action_ = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.q_next_ = tf.placeholder(shape=[None], dtype=tf.float32)

                    with tf.name_scope('train'):
                        action_OH = tf.one_hot(self.action_, self.action_size, dtype = tf.float32)
                        Q = tf.reduce_sum(tf.multiply(self.q, action_OH), axis = 1)
                        self.q_loss = tf.reduce_sum(tf.square(self.q_next_ - Q))

                    with tf.name_scope('local_grad'):
                        q_grads = tf.gradients(self.q_loss, self.q_vars)

                    # Sync with Global Network
                    with tf.name_scope('sync'):
                        # self.pull_op = self._build_pull(self.q_vars, self.global_network.q_vars)
                        self.train_ops, self.update_ops = self._build_push(q_grads, self.global_network.q_vars, self.optimizer, tau=0.95)

    def _build_q_network(self, input_, gps_, goal_, reuse=False):
        with tf.variable_scope('Q', reuse=reuse):
            net = Deep_layer.conv2d_pool(input_, [32,64,64], [5,3,2], [2,2,2], [1,1,1],
                                         padding='SAME', flatten=True)
            #gps_array = Deep_layer.fc(input_layer=gps_,
            #                          hidden_layers=[64, 64],
            #                          dropout=1.0,
            #                          scope='gps_proc')
            global_array = Deep_layer.fc(input_layer=goal_,
                                       hidden_layers=[64, 64],
                                       dropout=1.0,
                                       scope='goal_proc')
            net = tf.concat([net, global_array], 1)
            net = layers.fully_connected(net, 128)
            q = layers.fully_connected(net,
                                       self.action_size,
                                       weights_initializer=layers.xavier_initializer(),
                                       biases_initializer=tf.zeros_initializer(),
                                       activation_fn=None)

        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/Q')

        return q, q_vars

    def _build_policy_network(self, input_, gps_, goal_, reuse=False):
        with tf.variable_scope('actor', reuse=reuse):
            net = Deep_layer.conv2d_pool(input_, [32,64,64], [5,3,2], [2,2,2], [1,1,1],
                                         padding='SAME', flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64, 64],
                                      dropout=1.0,
                                      scope='gps_proc')
            goal_array = Deep_layer.fc(input_layer=goal_,
                                       hidden_layers=[64, 64],
                                       dropout=1.0,
                                       scope='goal_proc')
            net = tf.concat([net, gps_array, goal_array], 1)
            net = layers.fully_connected(net, 128)
            actor = layers.fully_connected(net,
                                           self.action_size,
                                           weights_initializer=layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer(),
                                           activation_fn=tf.nn.softmax)
        with tf.variable_scope('critic'):
            critic = layers.fully_connected(net,
                                            1,
                                            weights_initializer=layers.xavier_initializer(),
                                            biases_initializer=tf.zeros_initializer(),
                                            activation_fn=None)
            critic = tf.reshape(critic, [-1])


        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/actor')
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/critic')

        return actor, critic, a_vars, c_vars

    def _build_pull(self, local_vars, global_vars):
        with tf.name_scope('pull'):
            pull_op = [local_var.assign(glob_var) for local_var, glob_var in zip(local_vars, global_vars)]
        return pull_op

    def _build_push(self, grads, var, optimizer, tau=1.0):
        if self.explicit_policy:
            with tf.name_scope('push'):
                update_op = optimizer.apply_gradients(zip(grads, var))
            return update_op
        else:
            eta = 1-tau
            train_ops = optimizer.apply_gradients(zip(grads, self.q_vars))
            update_ops = [glob_var.assign(glob_var.value()*tau + local_var.value()*eta)
                            for local_var, glob_var in zip(var, self.global_network.q_vars)]
            return train_ops, update_ops

    # Update global network with local gradients
    # Choose Action
    def run_network(self, local_state, shared_state, goal_state):
        if self.explicit_policy:
            feed_dict = {self.state_input_: np.stack(local_state),
                         self.gps_state_: np.stack(shared_state),
                         self.goal_state_: np.stack(goal_state)}
            a_probs, critic = self.sess.run([self.actor, self.critic], feed_dict)
            return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic
        else:
            feed_dict = {self.state_input_: np.stack(local_state),
                         self.goal_state_: np.stack(goal_state)}
            q, pred = self.sess.run([self.q, self.predict], feed_dict)
            return pred, q

    def run_sample(self, local_state, shared_state, goal_state):
        if self.explicit_policy:
            feed_dict = {self.state_input_: np.stack(local_state),
                         self.gps_state_: np.stack(shared_state),
                         self.goal_state_: np.stack(goal_state)}
            a_probs = self.sess.run(self.actor, feed_dict)
            return a_probs
        else:
            feed_dict = {self.state_input_: np.stack(local_state),
                         self.goal_state_: np.stack(goal_state)}
            q = self.sess.run(self.q, feed_dict)
            return q

    def get_critic(self, local_state, shared_state, goal_state):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state),
                     self.goal_state_: np.stack(goal_state)}
        critic = self.sess.run(self.critic, feed_dict)
        return critic.tolist()

    def update_global(self, local_obs, gps_obs, action, advantage, goal, td_target, local_obs_1=None, q_next=None):
        if self.explicit_policy:
            feed_dict = {self.state_input_ : np.array(local_obs),
                         self.gps_state_ : np.array(gps_obs),
                         self.goal_state_ : np.array(goal),
                         self.action_ : np.array(action),
                         self.td_target_ : np.array(td_target),
                         self.advantage_ : np.array(advantage)}
            self.sess.run(self.update_ops, feed_dict)
            al, cl, etrpy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)
            return al, cl, etrpy
        else:
            feed_dict = {self.state_input_ : np.array(local_obs),
                         self.goal_state_ : np.array(goal),
                         self.action_ : np.array(action),
                         self.q_next_ : np.array(q_next)}
            self.sess.run(self.train_ops, feed_dict)
            self.sess.run(self.update_ops, feed_dict)
            q, ql = self.sess.run([self.q, self.q_loss], feed_dict)
            return q, ql


    def pull_global(self):
        self.sess.run(self.pull_op)
