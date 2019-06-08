import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utility.utils import store_args

from network.base import Deep_layer, Tensorboard_utility
from network.pg import Loss, Backpropagation


class HAC_subcontroller:
    """Actor Critic Network 

    This module contains building network and pipelines to sync with global network.
    Global network is expected to have same network structure.
    Actor Critic is implemented with convolution network and fully connected network.

    Todo:
        pass

    """
    @store_args
    def __init__(self,
                 local_state_shape,
                 shared_state_shape,
                 action_size,
                 scope,
                 strategy_size=3,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 entropy_beta=0,
                 critic_beta=1.0,
                 sess=None,
                 global_network=None,
                 global_step=None,
                 log_path=None):
        """ Initialize AC network and required parameters

        Keyword arguments:
            explicit_policy: If false, use single critic network and Q value for each action.

        Note:
            Any tensorflow holder is marked with underscore at the end of the name.
                ex) action holder -> action_
                    td_target holder -> td_target_
                - Also indicating that the value will not pass on backpropagation.

        TODO:

        """

        build_loss = Loss.softmax_cross_entropy_selection
        build_train = Backpropagation.asynch_pipeline

        with tf.variable_scope(scope):
            self.state_input_ = tf.placeholder(shape=local_state_shape, dtype=tf.float32, name='state')
            self.gps_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='gps_state')
            if global_network is not None:
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')


            # Build policy for each strategies
            self.actor_list, self.critic_list, self.a_vars_list, self.c_vars_list = [],[],[],[]
            self.actor_loss_list, self.critic_loss_list, self.entropy_list = [],[],[]
            self.pull_op_list, self.update_ops_list = [],[]
            for strat_id in range(strategy_size):
                actor, critic, a_vars, c_vars = self._build_policy_network(self.state_input_, self.gps_state_, policy_id=strat_id)
                self.actor_list.append(actor)
                self.critic_list.append(critic)
                self.a_vars_list.append(a_vars)
                self.c_vars_list.append(c_vars)

                if global_network is not None:
                    actor_loss, critic_loss, entropy = build_loss(actor, self.action_, self.advantage_, self.td_target_, critic, name_scope='loss_'+str(strat_id))
                    pull_op, update_ops = build_train(actor_loss, critic_loss,
                                                      a_vars, c_vars,
                                                      global_network.a_vars_list[strat_id], global_network.c_vars_list[strat_id],
                                                      lr_actor, lr_critic,
                                                      name_scope='sync_'+str(strat_id))
                    self.actor_loss_list.append(actor_loss)
                    self.critic_loss_list.append(critic_loss)
                    self.entropy_list.append(entropy)
                    self.pull_op_list.append(pull_op)
                    self.update_ops_list.append(update_ops)

            # Summarize
            if global_network is None:
                summaries = []
                for var in tf.trainable_variables(scope=scope):
                    summaries.append(tf.summary.histogram(var.name, var))
                self.merged_summary_op = tf.summary.merge(summaries)

    def _build_policy_network(self, input_, gps_, reuse=False, policy_id=0):
        policy_name = "policy_"+str(policy_id)
        critic_name = "critic_"+str(policy_id)
        with tf.variable_scope(policy_name, reuse=reuse):
            net = Deep_layer.conv2d_pool(input_, [32,64,64], [5,3,2], [2,2,2], [1,1,1],
                                         padding='SAME', flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64,64],
                                      dropout=1.0,
                                      scope='gps_proc')
            #net = tf.concat([net, gps_array], 1)
            net = layers.fully_connected(net, 128)
            actor = layers.fully_connected(net,
                                           self.action_size,
                                           weights_initializer=layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer(),
                                           activation_fn=tf.nn.softmax)
        with tf.variable_scope(critic_name, reuse=reuse):
            critic = layers.fully_connected(net,
                                            1,
                                            weights_initializer=layers.xavier_initializer(),
                                            biases_initializer=tf.zeros_initializer(),
                                            activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+policy_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+critic_name)

        return actor, critic, a_vars, c_vars


    # Update global network with local gradients
    # Choose Action
    def run_network(self, local_state, shared_state, strategy_id:int):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        a_probs, critic = self.sess.run([self.actor_list[strategy_id], self.critic_list[strategy_id]], feed_dict)
        return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic

    def run_sample(self, local_state, shared_state, strategy_id:int):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        a_probs = self.sess.run(self.actor_list[strategy_id], feed_dict)
        return a_probs

    def get_critic(self, local_state, shared_state, strategy_id:int):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        critic = self.sess.run(self.critic_list[strategy_id], feed_dict)
        return critic.tolist()

    def update_global(self, local_obs, gps_obs, action, advantage, td_target, strategy_id:int,
            log=False, writer=None):
        feed_dict = {self.state_input_ : np.array(local_obs),
                     self.gps_state_ : np.array(gps_obs),
                     self.action_ : np.array(action),
                     self.td_target_ : np.array(td_target),
                     self.advantage_ : np.array(advantage)}
        self.sess.run(self.update_ops_list[strategy_id], feed_dict)

        if log:
            assert self.global_step is not None, "global step is not passed to logger"
            assert writer is not None, "writer is not given"
            ops = [self.actor_loss_list[strategy_id],
                   self.critic_loss_list[strategy_id],
                   self.entropy_list[strategy_id] ]
            a_loss, c_loss, entropy = self.sess.run(ops, feed_dict)
            hist_summary = self.sess.run(self.global_network.merged_summary_op)
            step = self.sess.run(self.global_step)
            Tensorboard_utility.scalar_logger(f'train_summary/entropy_sid{strategy_id}', entropy, step, writer)
            Tensorboard_utility.scalar_logger(f'train_summary/a_loss_sid{strategy_id}', a_loss, step, writer)
            Tensorboard_utility.scalar_logger(f'train_summary/c_loss_sid{strategy_id}', c_loss, step, writer)
            Tensorboard_utility.histogram_logger(hist_summary, step, writer)


    def pull_global(self, strategy_id:int):
        self.sess.run(self.pull_op_list[strategy_id])

    def pull_global_all(self):
        self.sess.run(self.pull_op_list)


class HAC_meta_controller:
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
                 sess=None,
                 global_network=None):
        with tf.variable_scope(scope):
            self.state_input_ = tf.placeholder(shape=local_state_shape, dtype=tf.float32, name='state')
            self.gps_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='gps_state')
            if global_network is not None:
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_holder')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_holder')

            self.critic_optimizer = tf.train.AdamOptimizer(self.lr_critic, name='Adam_critic')
            self.actor_optimizer = tf.train.AdamOptimizer(self.lr_actor, name='Adam_actor')

            # Build policy for each strategies
            self.actor, self.critic, self.a_vars, self.c_vars = self._build_policy_network(self.state_input_, self.gps_state_)

            if global_network is not None:
                self.actor_loss, self.critic_loss, self.entropy = self._build_loss(self.actor, self.action_, self.advantage_, self.td_target_, self.critic)
                self.pull_op, self.update_ops = self._build_train(self.actor_loss, self.critic_loss,
                                                                  self.a_vars, self.c_vars,
                                                                  self.global_network.a_vars, self.global_network.c_vars)

    def _build_loss(self, actor, action, advantage, td_target, critic):
        with tf.name_scope('train'):
            # Critic (value) Loss
            td_error = td_target - critic
            entropy = -tf.reduce_mean(actor * tf.log(actor), name='entropy')
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size)
            obj_func = tf.log(tf.reduce_sum(actor * action_OH, 1))
            exp_v = obj_func * advantage
            actor_loss = tf.reduce_mean(-exp_v, name='actor_loss') - self.entropy_beta * entropy

        return actor_loss, critic_loss, entropy

    def _build_train(self, actor_loss, critic_loss,
                        a_vars, c_vars, a_targ_vars, c_targ_vars):
        def _build_pull(local_vars, global_vars):
            with tf.name_scope('pull'):
                pull_op = [local_var.assign(glob_var) for local_var, glob_var in zip(local_vars, global_vars)]
            return pull_op

        def _build_push(grads, var, optimizer, tau=1.0):
            with tf.name_scope('push'):
                update_op = optimizer.apply_gradients(zip(grads, var))
            return update_op

        with tf.name_scope('local_grad'):
            a_grads = tf.gradients(actor_loss, a_vars)
            c_grads = tf.gradients(critic_loss, c_vars)

        # Sync with Global Network
        with tf.name_scope('sync'):
            pull_a_vars_op = _build_pull(a_vars, a_targ_vars)
            pull_c_vars_op = _build_pull(c_vars, c_targ_vars)
            pull_op = tf.group(pull_a_vars_op, pull_c_vars_op)

            update_a_op = _build_push(a_grads, a_targ_vars, self.actor_optimizer)
            update_c_op = _build_push(c_grads, c_targ_vars, self.critic_optimizer)
            update_ops = tf.group(update_a_op, update_c_op)

        return pull_op, update_ops

    def _build_policy_network(self, input_, gps_, reuse=False):
        policy_name = "policy"
        critic_name = "critic"
        with tf.variable_scope(policy_name, reuse=reuse):
            net = Deep_layer.conv2d_pool(input_, [32,64,64], [5,3,2], [2,2,2], [1,1,1],
                                         padding='SAME', flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64, 64],
                                      dropout=1.0,
                                      scope='gps_proc')
            net = tf.concat([net, gps_array], 1)
            net = layers.fully_connected(net, 64)
            actor = layers.fully_connected(net,
                                           self.action_size,
                                           weights_initializer=layers.xavier_initializer(),
                                           biases_initializer=tf.zeros_initializer(),
                                           activation_fn=tf.nn.softmax)
        with tf.variable_scope(critic_name, reuse=reuse):
            critic = layers.fully_connected(net,
                                            1,
                                            weights_initializer=layers.xavier_initializer(),
                                            biases_initializer=tf.zeros_initializer(),
                                            activation_fn=None)
            critic = tf.reshape(critic, [-1])

        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+policy_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/'+critic_name)

        return actor, critic, a_vars, c_vars

    # Update global network with local gradients
    # Choose Action
    def run_network(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        a_probs, critic = self.sess.run([self.actor, self.critic], feed_dict)
        return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs], critic

    def run_sample(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        a_probs = self.sess.run(self.actor_list, feed_dict)
        return a_probs

    def get_critic(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        critic = self.sess.run(self.critic, feed_dict)
        return critic.tolist()

    def update_global(self, local_obs, gps_obs, action, advantage, td_target):
        feed_dict = {self.state_input_ : np.array(local_obs),
                     self.gps_state_ : np.array(gps_obs),
                     self.action_ : np.array(action),
                     self.td_target_ : np.array(td_target),
                     self.advantage_ : np.array(advantage)}
        self.sess.run(self.update_ops, feed_dict)
        a_loss, c_loss, entropy = self.sess.run([self.actor_loss, self.critic_loss, self.entropy], feed_dict)
        return a_loss, c_loss, entropy

    def pull_global_all(self):
        self.sess.run(self.pull_op)

class HAC_meta_controller_DQN:
    @store_args
    def __init__(self,
                 local_state_shape,
                 shared_state_shape,
                 action_size,
                 scope,
                 lr_q=1e-4,
                 sess=None,
                 target_network=None):
        with tf.variable_scope(scope):
            # global Network
            # Build actor and critic network weights. (global network does not need training sequence)
            self.state_input_ = tf.placeholder(shape=local_state_shape, dtype=tf.float32, name='state')
            self.gps_state_ = tf.placeholder(shape=shared_state_shape, dtype=tf.float32, name='gps_state')

            # get the parameters of actor and critic networks
            self.q, self.q_vars = self._build_q_network(self.state_input_, self.gps_state_)
            self.predict = tf.argmax(self.q, axis=-1)

            if target_network is not None:
                self.optimizer = tf.train.AdamOptimizer(self.lr_q, name='Adam_critic')

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
                    # self.pull_op = self._build_pull(self.q_vars, self.target_network.q_vars)
                    self.train_ops, self.update_ops = self._build_push(q_grads, self.target_network.q_vars, self.optimizer, tau=0.95)

    def _build_q_network(self, input_, gps_, reuse=False):
        with tf.variable_scope('Q', reuse=reuse):
            net = Deep_layer.conv2d_pool(input_, [32,64,64], [5,3,2], [2,2,2], [1,1,1],
                                         padding='SAME', flatten=True)
            gps_array = Deep_layer.fc(input_layer=gps_,
                                      hidden_layers=[64, 64],
                                      dropout=1.0,
                                      scope='gps_proc')
            net = tf.concat([net, gps_array], 1)
            net = layers.fully_connected(net, 128)
            q = layers.fully_connected(net,
                                       self.action_size,
                                       weights_initializer=layers.xavier_initializer(),
                                       biases_initializer=tf.zeros_initializer(),
                                       activation_fn=None)

        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'/Q')

        return q, q_vars

    def _build_pull(self, local_vars, global_vars):
        with tf.name_scope('pull'):
            pull_op = [local_var.assign(glob_var) for local_var, glob_var in zip(local_vars, global_vars)]
        return pull_op

    def _build_push(self, grads, var, optimizer, tau=1.0):
        eta = 1-tau
        train_ops = optimizer.apply_gradients(zip(grads, self.q_vars))
        update_ops = [glob_var.assign(glob_var.value()*tau + local_var.value()*eta)
                        for local_var, glob_var in zip(var, self.target_network.q_vars)]
        return train_ops, update_ops

    # Update global network with local gradients
    # Choose Action
    def run_network(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state)}
        q, pred = self.sess.run([self.q, self.predict], feed_dict)
        return pred, q

    def run_sample(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state)}
        q = self.sess.run(self.q, feed_dict)
        return q

    def get_critic(self, local_state, shared_state):
        feed_dict = {self.state_input_: np.stack(local_state),
                     self.gps_state_: np.stack(shared_state)}
        critic = self.sess.run(self.critic, feed_dict)
        return critic.tolist()

    def update_global(self, local_obs, gps_obs, action, advantage, td_target, local_obs_1=None, q_next=None):
        feed_dict = {self.state_input_ : np.array(local_obs),
                     self.action_ : np.array(action),
                     self.q_next_ : np.array(q_next)}
        self.sess.run(self.train_ops, feed_dict)
        self.sess.run(self.update_ops, feed_dict)
        q, ql = self.sess.run([self.q, self.q_loss], feed_dict)
        return q, ql

    def pull_global(self):
        self.sess.run(self.pull_op)

