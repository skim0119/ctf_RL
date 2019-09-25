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
from method.base import put_ctf_state_on_grid


#from network.attention_ctf import build_network
from network.attention import self_attention
from network.model_V2 import build_network as build_network
from network.model_V3 import build_network as build_network_V3

class Loss:
    """Loss

    Build function for commonly used loss functions for Policy gradient

    The default is the 'softmax cross-entropy selection' for actor loss and 'TD-difference' for critic error

    """
    @staticmethod
    def _log(val):
        #return tf.debugging.check_numerics(tf.log(val),'log nan found')
        return tf.log(tf.clip_by_value(val, 1e-10, 1.0))

    @staticmethod
    def ppo(policy, log_prob, old_log_prob,
            action, advantage,
            td_target, critic,
            entropy_beta=0.001, critic_beta=0.5,
            eps=0.2,
            name_scope='loss'):
        with tf.name_scope(name_scope):
            # Entropy
            entropy = -tf.reduce_mean(policy * Loss._log(policy), name='entropy')

            # Critic Loss
            td_error = td_target - critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_size = tf.shape(policy)[1]
            action_OH = tf.one_hot(action, action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(log_prob * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_prob * action_OH, 1)

            # Clipped surrogate function
            ratio = tf.exp(log_prob - old_log_prob)
            #ratio = log_prob / old_log_prob
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-eps, 1+eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            if entropy_beta != 0:
                actor_loss = actor_loss - entropy * entropy_beta
            if critic_beta != 0:
                actor_loss = actor_loss + critic_loss * critic_beta

        return actor_loss, critic_loss, entropy


class PPO(a3c):
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
            loss = Loss.ppo
            backprop = Backpropagation.selfupdate

            with tf.variable_scope(scope):
                self._build_placeholder(in_size)
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_holder')

                # get the parameters of actor and critic networks
                self.logits, self.actor, self.critic, self.a_vars, self.c_vars = self._build_network(self.state_input)
                self.logits = tf.nn.log_softmax(self.logits) # Use log probability for PPO

                # Local Network
                train_args = (self.action_, self.advantage_, self.td_target_)
                loss = loss(self.actor, self.logits, self.old_logits_,
                        *train_args, self.critic, entropy_beta=entropy_beta)
                self.actor_loss, self.critic_loss, self.entropy = loss

                self.update_ops, self.gradients = backprop(
                    self.actor_loss, self.critic_loss,
                    self.a_vars, self.c_vars,
                    lr_actor, lr_critic,
                    return_gradient=True,
                    single_loss=True
                )

            # Summary
            grad_summary = []
            for tensor, grad in zip(self.a_vars+self.c_vars, self.gradients):
                grad_summary.append(tf.summary.histogram("%s-grad" % tensor.name, grad))
            self.merged_grad_summary_op = tf.summary.merge(grad_summary)
            self.merged_summary_op = self._build_summary(self.a_vars + self.c_vars)

    def run_network(self, states):
        feed_dict = {self.state_input: states}
        a_probs, critics, logits = self.sess.run([self.actor, self.critic, self.logits], feed_dict)
        actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
        return actions, critics, logits

    def update_global(self, state_input, action, td_target, advantage, old_logit, global_episodes, writer=None, log=False):
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

    def _build_network(self, input_hold):
        encoder_name = self.scope + '/encoder'
        actor_name = self.scope + '/actor'
        critic_name = self.scope + '/critic'

        image_summary = []
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        # Feature encoder
        with tf.variable_scope('encoder'):
            feature, _layers = build_network(input_hold)
            add_image(_layers['input'], '1_input', X=7)
            add_image(_layers['sepCNN1'], '2_sepCNN')
            add_image(_layers['attention'], '3_attention')
            add_image(_layers['NLNN'], '4_nonlocal')
            add_image(_layers['CNN1'], '5_CNN')
            add_image(_layers['CNN2'], '6_CNN')

        # Actor
        with tf.variable_scope('actor'):
            logits = layers.fully_connected(
                feature, self.action_size,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)

        # Critic
        with tf.variable_scope('critic'):
            critic = layers.fully_connected(
                feature, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        # Collect Variable
        e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=encoder_name)
        a_vars = e_vars+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        # Collect Summary
        self.cnn_summary = tf.summary.merge(image_summary)

        # Visualization
        #self.feature_static = _layers['sepCNN1']
        #self.feature_dynamic = _layers['attention']
        #self.feature_attention = _layers['NLNN']
        #labels = tf.one_hot(self.action_, self.action_size, dtype=tf.float32)
        #yc = tf.reduce_sum(logits * labels, axis=1)
        #self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        #self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        #self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]

        return logits, actor, critic, a_vars, c_vars


class PPO_multimodes(a3c):
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
        tau=1.0,
        num_mode=None,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."
        assert num_mode is not None, "Number of mode must be specified"
        if global_network is None:
            global_network = self

        with self.sess.as_default(), self.sess.graph.as_default():
            #loss = kwargs.get('loss', Loss.softmax_cross_entropy_selection)
            backprop = Backpropagation.selfupdate

            with tf.variable_scope(scope):
                self._build_placeholder(in_size)
                self.bandit_action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='bandit_hold')
                self.old_bandit_logits_ = tf.placeholder(shape=[None, num_mode], dtype=tf.float32, name='old_bandit_logit_holder')
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_holder')

                self.logits, self.actor, self.critic, self.a_vars, self.c_vars = self._build_network(self.state_input)

                train_args = (self.action_, self.advantage_, self.td_target_)
                self.actor_loss = []
                self.critic_loss = []
                self.entropy = []
                self.update_ops = []
                self.gradients = []
                for option in range(num_mode):
                    loss = Loss.ppo(
                            self.actor[option],
                            self.logits[option],
                            self.old_logits_,
                            *train_args,
                            self.critic[option],
                            entropy_beta=entropy_beta,
                            name_scope='loss{}'.format(option)
                        )
                    actor_loss, critic_loss, entropy = loss

                    update_ops, gradients = backprop(
                        actor_loss, critic_loss,
                        self.a_vars[option], self.c_vars[option],
                        lr_actor, lr_critic,
                        return_gradient=True,
                        single_loss=True,
                        name_scope='sync{}'.format(option)
                    )

                    self.actor_loss.append(actor_loss)
                    self.critic_loss.append(critic_loss)
                    self.entropy.append(entropy)
                    self.update_ops.append(update_ops)
                    self.gradients.append(gradients)

                # Loss for bandit
                train_args = (self.bandit_action_, self.advantage_, self.td_target_)
                loss = Loss.ppo(
                        self.actor[-1],
                        self.logits[-1],
                        self.old_bandit_logits_,
                        *train_args,
                        self.critic[-1],
                        entropy_beta=entropy_beta,
                        name_scope='meta_loss'
                    )
                actor_loss, critic_loss, entropy = loss

                update_ops, gradients = backprop(
                    actor_loss, critic_loss,
                    self.a_vars[-1], self.c_vars[-1],
                    lr_actor, lr_critic,
                    return_gradient=True,
                    single_loss=True,
                    name_scope='meta_sync'
                )

                self.actor_loss.append(actor_loss)
                self.critic_loss.append(critic_loss)
                self.entropy.append(entropy)
                self.update_ops.append(update_ops)
                self.gradients.append(gradients)

            # Summary (excluded: gradient vanishing ratio is enough to see)
            '''
            grad_summary = []
            for tensor, grad in zip(self.a_vars+self.c_vars, self.gradients):
                grad_summary.append(tf.summary.histogram("%s-grad" % tensor.name, grad))
            self.merged_grad_summary_op = tf.summary.merge(grad_summary)
            self.merged_summary_op = self._build_summary(self.a_vars + self.c_vars)
            '''
        self.threshold = 1.0 # Have threshhold increase while training. Make it the average maximum?


    def run_network(self, states, idx=None):
        feed_dict = {self.state_input: states}
        a_probs, critics, logits = self.sess.run([self.actor[idx], self.critic[idx], self.logits[idx]], feed_dict)
        actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
        return actions, critics, logits

    def run_network_all(self, states):
        feed_dict = {self.state_input: states}
        ops = [self.actor[0], self.critic[0], self.logits[0], self.actor[1], self.critic[1], self.logits[1], self.actor[2], self.critic[2], self.logits[2]]
        res = self.sess.run(ops, feed_dict)
        a_probs1, critics1, logits1, a_probs2, critics2, logits2, a_probs3, critics3, logits3 = res
        actions1 = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs1])
        actions2 = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs2])
        actions3 = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs3])
        return actions1, critics1, logits1, actions2, critics2, logits2, actions3, critics3, logits3



    def initiate_confid(self, n, use_confid=0, confid_params=[0.05, 0.10, 0.01]):
        """
        Overall initialization of confidence. To be run once when network is started.
        use_confid      : Decides which confidence metric to be used. (Also capable of toggling if a Fixed Step Scheme is Used.)
                        0 - None; 1 - Confidence; 2 - Threshold; 5 - Fixed Step;
        confid_params   : Parameters used in the confidence calculations.
        """
        self.use_confid = use_confid
        self.confid_params = confid_params

        if self.use_confid == 1:
            self.entering_confids = np.ones(n)
            self.playing_mode = np.zeros(n, dtype=int)

        if self.use_confid == 2:
            self.threshold = np.zeroes(n)


        if self.use_confid == 5:
            self.fixed_length = int(confid_params[0])
            self.fixed_length_counter = 0

    def initiate_episode_confid(self, n):
        """
        Confidence initialization steps to be perfromed each episode.
        """
        if self.use_confid == 1: #Need to reset the Playing Mode and entering Confidence every step.
            self.entering_confids = np.ones(n)
            self.playing_mode = np.zeros(n, dtype=int)


    def run_network_with_bandit(self, states, bandit_prob):
        """ """

        feed_dict = {self.state_input: states}
        ops = self.actor[:-1] + self.critic[:-1] + self.logits[:-1]
        res = self.sess.run(ops, feed_dict)
        a_probs, res = res[:len(self.actor)-1], res[len(self.actor)-1:]
        critics, logits = res[:len(self.critic)-1], res[len(self.critic)-1:]

        try:
            bandit_action = np.array([np.random.choice(self.num_mode, p=prob / sum(prob)) for prob in bandit_prob])
        except Exception as e:
            print(e)
            print(bandit_prob)
            raise Exception('probability nan')

        # Confidence
        if self.use_confid == 1: # Confidence Metric
            confids = -np.mean(bandit_prob * np.log(bandit_prob), axis=1)
            for i in range(len(self.entering_confids)):
                confid = confids[i]
                old_confid = self.entering_confids[i]
                if confid < old_confid: # compare inverse entropy
                    self.entering_confids[i] = confid
                    # print(self.playing_mode[i] , bandit_action[i])
                    self.playing_mode[i] = bandit_action[i]
                else:
                    self.entering_confids[i] = np.maximum(old_confid + self.confid_params[0], self.confid_params[1])

            bandit_action = self.playing_mode

        elif self.use_confid == 2: #Threshold Metric
            confids = -np.mean(bandit_prob * np.log(bandit_prob), axis=1)
            for i in range(len(self.entering_confids)):
                confid = confids[i]
                if  confid < self.threshold-self.confid_params[2]: # compare inverse entropy
                    self.playing_mode[i] = bandit_action[i]

                self.threshold = np.maximum((1-self.confid_params[0])*self.threshold + self.confid_params[0]*confid,self.confid_params[1])
                bandit_action = self.playing_mode

        elif self.use_confid == 5: #Fixed Step
            if self.fixed_length_counter == 0:
                self.playing_mode = bandit_action
                self.fixed_length_counter = self.fixed_length
            else:
                bandit_action = self.playing_mode
                self.fixed_length_counter -= 1


        actions = np.array([np.random.choice(self.action_size, p=a_probs[mod][idx] / sum(a_probs[mod][idx])) for idx, mod in enumerate(bandit_action)])

        critics = [critics[mod][idx] for idx, mod in enumerate(bandit_action)]
        logits = [logits[mod][idx] for idx, mod in enumerate(bandit_action)]
        return actions, critics, logits, bandit_action

    def update_global(self, state_input, action, td_target, advantage, old_logit, global_episodes, writer=None, log=False, idx=None):
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}
        self.sess.run(self.update_ops[idx], feed_dict)

        ops = [self.actor_loss[idx], self.critic_loss[idx], self.entropy[idx]]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        if log:
            log_ops = [self.cnn_summary]
            summaries = self.sess.run(log_ops, feed_dict)
            for summary in summaries:
                writer.add_summary(summary, global_episodes)
            summary = tf.Summary()
            summary.value.add(tag='summary/sub/actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/sub/critic_loss', simple_value=closs)
            summary.value.add(tag='summary/sub/entropy', simple_value=entropy)
            summary.value.add(tag='summary/sub/threshold', simple_value=self.threshold)

            # Check vanish gradient
            grads = self.sess.run(self.gradients[idx], feed_dict)
            total_counter = 0
            vanish_counter = 0
            for grad in grads:
                total_counter += np.prod(grad.shape)
                vanish_counter += (np.absolute(grad)<1e-8).sum()
            summary.value.add(tag='summary/sub/grad_vanish_rate', simple_value=vanish_counter/total_counter)

            writer.add_summary(summary,global_episodes)
            writer.flush()

    def update_bandit(self, state_input, action, td_target, advantage, old_logit, global_episodes, writer=None, log=False):
        feed_dict = {self.state_input: state_input,
                     self.bandit_action_: action,
                     self.td_target_: td_target,
                     self.advantage_: advantage,
                     self.old_bandit_logits_: old_logit}
        idx = -1
        self.sess.run(self.update_ops[idx], feed_dict)

        ops = [self.actor_loss[idx], self.critic_loss[idx], self.entropy[idx]]
        aloss, closs, entropy = self.sess.run(ops, feed_dict)

        if log:
            log_ops = [self.cnn_summary]
            summaries = self.sess.run(log_ops, feed_dict)
            for summary in summaries:
                writer.add_summary(summary, global_episodes)
            summary = tf.Summary()
            summary.value.add(tag='summary/bandit_actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/bandit_critic_loss', simple_value=closs)
            summary.value.add(tag='summary/bandit_entropy', simple_value=entropy)
            summary.value.add(tag='summary/threshold', simple_value=self.threshold)

            # Check vanish gradient
            grads = self.sess.run(self.gradients[idx], feed_dict)
            total_counter = 0
            vanish_counter = 0
            for grad in grads:
                total_counter += np.prod(grad.shape)
                vanish_counter += (np.absolute(grad)<1e-8).sum()
            summary.value.add(tag='summary/grad_vanish_rate_bandit', simple_value=vanish_counter/total_counter)

            writer.add_summary(summary,global_episodes)
            writer.flush()

    def _build_network(self, input_hold):
        encoder_name = self.scope + '/encoder'

        image_summary = []
        def add_image(net, name, Y=-1, X=8):
            grid = put_channels_on_grid(net[0], Y, X)
            image_summary.append(tf.summary.image(name, grid, max_outputs=1))

        # Encoder
        with tf.variable_scope('encoder'):
            feature, _layers = build_network(input_hold)
            add_image(_layers['input'], '1_input', X=7)
            add_image(_layers['sepCNN1'], '2_sepCNN')
            add_image(_layers['attention'], '3_attention')
            add_image(_layers['NLNN'], '4_nonlocal')
            add_image(_layers['CNN1'], '5_CNN')
            add_image(_layers['CNN2'], '6_CNN')

        # Actor-Critic
        logit_list = []
        actor_list = []
        critic_list = []
        a_vars_list = []
        c_vars_list = []
        for option in range(self.num_mode):
            actor_name = self.scope + '/actor_{}'.format(option)
            critic_name = self.scope + '/critic_{}'.format(option)
            with tf.variable_scope('actor_{}'.format(option)):
                logits = layers.fully_connected(
                    feature, self.action_size,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    activation_fn=None)
                actor = tf.nn.softmax(logits)
                log_softmax = tf.nn.log_softmax(logits)

            # Critic
            with tf.variable_scope('critic_{}'.format(option)):
                critic = layers.fully_connected(
                    feature, 1,
                    weights_initializer=layers.xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    activation_fn=None)
                critic = tf.reshape(critic, [-1])

            # Collect Variable
            e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=encoder_name)
            a_vars = e_vars+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
            c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

            logit_list.append(log_softmax)
            actor_list.append(actor)
            critic_list.append(critic)
            a_vars_list.append(a_vars)
            c_vars_list.append(c_vars)

        # Bandit
        actor_name = self.scope + '/bandit'
        critic_name = self.scope + '/bandit_critic'
        with tf.variable_scope('bandit'):
            logits = layers.fully_connected(
                feature, self.num_mode,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            actor = tf.nn.softmax(logits)
            log_softmax = tf.nn.log_softmax(logits)

        # Critic
        with tf.variable_scope('bandit_critic'):
            critic = layers.fully_connected(
                feature, 1,
                weights_initializer=layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn=None)
            critic = tf.reshape(critic, [-1])

        # Collect Variable
        e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=encoder_name)
        a_vars = e_vars+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=actor_name)
        c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=critic_name)

        logit_list.append(log_softmax)
        actor_list.append(actor)
        critic_list.append(critic)
        a_vars_list.append(a_vars)
        c_vars_list.append(c_vars)

        # Collect Summary
        self.cnn_summary = tf.summary.merge(image_summary)

        # Visualization (Gradcam)
        '''
        self.feature_static = _layers['sepCNN1']
        self.feature_dynamic = _layers['attention']
        self.feature_attention = _layers['NLNN']
        labels = tf.one_hot(self.action_, self.action_size, dtype=tf.float32)
        yc = tf.reduce_sum(logits * labels, axis=1)
        self.conv_layer_grad_dynamic = tf.gradients(yc, self.feature_dynamic)[0]
        self.conv_layer_grad_static = tf.gradients(yc, self.feature_static)[0]
        self.conv_layer_grad_attention = tf.gradients(yc, self.feature_attention)[0]
        '''

        return logit_list, actor_list, critic_list, a_vars_list, c_vars_list
