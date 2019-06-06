import random
import numpy as np
from collections import defaultdict

import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import tensorflow as tf

from utility.utils import retrace, store_args

# TODO:
'''
- Instead of limiting number of policy network for local,
    just create full list of network,
- Reduce number of operation node per graph.
- Remove unnecessary network structures
'''


# implementation for decentralized action and centralized critic
class Critic_sharing():
    @store_args
    def __init__(self,
                 in_size,
                 action_size,
                 num_agent,
                 scope,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm=0,
                 entropy_beta=0.001,
                 sess=None,
                 global_network=None,
                 num_policy_pool=4,
                 allow_same_policy=False,
                 ):
        """ Initialize AC network and required parameters

        Keyword arguments:
        in_size - network state input shape
        action_size - action space size (int)
        num_agent - number of agent
        scope - name scope of the network. (special for 'global')
        lr_actor - learning rate for actor
        lr_critic - learning rate for critic
        grad_clip_norm - normalize gradient clip (0 for no clip)
        entropy_beta - entropy weight
        sess - tensorflow session
        global_network - global network
        num_policy_pool - number of policy population
        allow_same_policy - if true, allow two agency to have shared policy
        """

        # Configurations and Parameters
        self.is_global = (scope == 'global')
        self.retrace_lambda = 0.202

        # Learning Rate Variables and Parameters
        with tf.variable_scope(scope):
            # Optimizer
            self.a_opt = tf.train.AdamOptimizer(self.lr_actor)
            self.c_opt = tf.train.AdamOptimizer(self.lr_critic)

            # Network Structure
            # Build actor network weights.

            # Set policy network
            # Unshared Policy
            self.state_inputs_ = []
            self.actors = []
            self.a_vars = []
            for policy_id in range(self.num_policy_pool):  # number of policy
                state_input, actor, a_var = self._build_actor_network(policy_id=policy_id)

                self.state_inputs_.append(state_input)
                self.actors.append(actor)
                self.a_vars.append(a_var)

            # Set critic network
            # Shared Critic
            self.critic_state_, self.critic, self.c_var = self._build_critic_network()

            # Local Network (Trainer)
            if not self.is_global:
                self.policy_index = self.select_policy()
                self._build_actor_loss()
                self._build_critic_loss()
                self._build_gradient()
                self._build_pipeline()

    def _build_actor_network(self, scope='actor', policy_id=""):
        """ policy network """
        scope = scope + str(policy_id)

        state_input = tf.placeholder(shape=self.in_size, dtype=tf.float32, name='ac_state_hold' + str(policy_id))

        with tf.variable_scope(scope):
            net = layers.conv2d(state_input, 16, [3, 3], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 32, [2, 2], activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.flatten(net)

            net = layers.fully_connected(net, self.action_size,
                                         activation_fn=tf.nn.softmax)

        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)

        return state_input, net, vars_list

    def _build_critic_network(self, scope='critic', critic_id=""):
        """ Common shared critic """
        scope = scope + str(critic_id)

        state_input = tf.placeholder(shape=self.in_size, dtype=tf.float32, name='cr_state_hold' + str(critic_id))

        with tf.variable_scope(scope):
            net = layers.conv2d(state_input, 16, [3, 3],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 32, [2, 2],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.flatten(net)
            net = layers.fully_connected(net, 1,
                                         activation_fn=None)
            net = tf.reshape(net, (-1,))

        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)

        return state_input, net, vars_list

    def _build_actor_loss(self):
        # Actor Loss
        # Placeholders: pipeline for values
        self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
        self.adv_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
        self.retrace_ = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_hold')

        self.actor_losses = []
        self.entropies = []
        with tf.name_scope('Actor_Loss'):
            for actor in self.actors:
                entropy = -tf.reduce_mean(actor * tf.log(actor))
                action_OH = tf.one_hot(self.action_, self.action_size)
                obj_func = tf.log(tf.reduce_sum(actor * action_OH, 1))
                exp_v = obj_func * self.adv_ * self.retrace_ + self.entropy_beta * entropy
                actor_loss = tf.reduce_mean(-exp_v)

                self.actor_losses.append(actor_loss)
                self.entropies.append(entropy)

    def _build_critic_loss(self):
        # Make sure critic can get the require result
        with tf.name_scope('Critic_Loss'):
            self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
            self.retrace_prod_ = tf.placeholder(shape=[None], dtype=tf.float32, name='sample_ratio_prod_hold')

            td_error = self.td_target_ - self.critic
            self.critic_loss = tf.reduce_mean(tf.square(td_error) * self.retrace_prod_, name='critic_loss')

    def _build_gradient(self):
        # Gradient
        self.actor_grads = []
        with tf.name_scope('gradient'):
            for aid in range(self.num_policy_pool):
                a_loss = self.actor_losses[aid]
                a_var = self.a_vars[aid]
                with tf.variable_scope('agent' + str(aid)):
                    a_grad = tf.gradients(a_loss, a_var)
                self.actor_grads.append(a_grad)
            self.critic_grad = tf.gradients(self.critic_loss, self.c_var)

    def _build_pipeline(self):
        # Pull global weights to local weights
        with tf.name_scope('pull'):
            self.pull_a_ops = []
            for lVars, gVars in zip(self.a_vars, self.global_network.a_vars):
                self.pull_a_ops.append([lVar.assign(gVar) for lVar, gVar in zip(lVars, gVars)])
            self.pull_c_op = [lVar.assign(gVar) for lVar, gVar in zip(self.c_var, self.global_network.c_var)]

        # Push local weights to global weights
        with tf.name_scope('push'):
            self.update_a_ops = []
            for lGrads, gVars in zip(self.actor_grads, self.global_network.a_vars):
                self.update_a_ops.append(self.a_opt.apply_gradients(zip(lGrads, gVars)))
            self.update_c_op = self.c_opt.apply_gradients(zip(self.critic_grad, self.global_network.c_var))

    def update_full(self, states, actions, advs, td_targets, beta_policies):
        # Complete update for actor policies and critic
        # All parameters are given for each agents
        a_loss, c_loss = [], []
        for idx, policy_id in enumerate(self.policy_index):
            s, a, adv, td, beta = states[idx], actions[idx], advs[idx], td_targets[idx], beta_policies[idx]
            if s is None:
                continue
            # Compute retrace weight
            feed_dict = {self.global_network.state_inputs_[policy_id]: np.stack(s)}
            soft_prob = self.sess.run(self.global_network.actors[policy_id], feed_dict)
            target_policy = np.array([ar[act] for ar, act in zip(soft_prob, a)])
            retrace_weight = retrace(target_policy, beta, self.retrace_lambda)
            retrace_prod = np.cumprod(retrace_weight)

            # Update specific policy
            feed_dict = {self.state_inputs_[policy_id]: np.stack(s),
                         self.action_: a,
                         self.adv_: adv,
                         self.retrace_: retrace_weight}
            loss, _ = self.sess.run([self.actor_losses[policy_id], self.update_a_ops[policy_id]], feed_dict)
            a_loss.append(loss)

            # Update critic
            feed_dict = {self.critic_state_: np.stack(s),
                         self.td_target_: td,
                         self.retrace_prod_: retrace_prod}
            loss, _ = self.sess.run([self.critic_loss, self.update_c_op], feed_dict)
            c_loss.append(loss)

        return a_loss, np.mean(c_loss)

    def pull_global(self, complete_pull=False):
        if complete_pull:
            self.sess.run([self.pull_a_op, self.pull_c_op])
        else:
            actor_ops = [self.pull_a_ops[i] for i in self.policy_index]
            self.sess.run([actor_ops, self.pull_c_op])

    # Return action and critic
    def get_action_critic(self, states):
        action_prob = []
        critic_list = []
        for agent_id, policy_id in enumerate(self.policy_index):
            # Construct feed dictionary
            state = states[agent_id]
            feed_dict = {self.state_inputs_[policy_id]: np.expand_dims(state, axis=0),
                         self.critic_state_: np.expand_dims(state, axis=0)}

            a, c = self.sess.run([self.actors[policy_id], self.critic], feed_dict)
            action_prob.append(a[0])
            critic_list.append(c[0])

        action = [np.random.choice(self.action_size, p=prob / sum(prob)) for prob in action_prob]

        return action, action_prob, critic_list

    # Policy Random Pool
    def select_policy(self):
        assert not self.is_global
        if self.allow_same_policy:
            policy_index = random.choices(range(self.num_policy_pool), k=self.num_agent)
        else:
            policy_index = random.sample(range(self.num_policy_pool), k=self.num_agent)
        return policy_index
