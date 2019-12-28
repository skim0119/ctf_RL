import tensorflow as tf
import tensorflow.contrib.layers as layers

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

from network.model_lstm import PPO_LSTM_V1

class Encoder(tf.keras.Model):
    @store_args
    def __init__(self, action_size=5, trainable=True, lr=1e-4, eps=0.2, entropy_beta=0.01, critic_beta=0.5, name='PPO'):
        super(PPO_LSTM_V1, self).__init__(name=name)

        # Feature Encoder
        conv1 = layers.SeparableConv2D(
                filters=16,
                kernel_size=5,
                strides=3,
                padding='valid',
                depth_multiplier=2,
                activation='relu',
            )
        self.td_conv1 = layers.TimeDistributed(conv1, input_shape=(None,4,39,39,6))

        conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu')
        self.td_conv2 = layers.TimeDistributed(conv2)

        flat = layers.Flatten()
        self.td_flat  = layers.TimeDistributed(flat)

        dense1 = layers.Dense(units=256, activation='relu')
        self.td_dense1 = layers.TimeDistributed(dense1)

        self.lstm1 = layers.LSTM(256, return_state=True)

        # Actor
        self.actor_dense1 = layers.Dense(action_size)
        self.softmax = layers.Activation('softmax')

        # Critic
        self.critic_dense1 = layers.Dense(5, activation='relu')
        self.critic_dense2 = layers.Dense(1)

    def call(self, inputs):
        # state_input : [None, keepframe, 39, 39, 6]
        # prev_action : [None, 1]
        # prev_reward : [None, 1]
        # hidden : [[None, 256], [None, 256]]
        state_input, prev_action, prev_reward, hidden = inputs

        net = state_input
        net = self.td_conv1(net)
        net = self.td_conv2(net)
        net = self.td_flat(net)
        net = self.td_dense1(net)
        net = tf.concat([net, prev_action, prev_reward], axis=2)
        #net, state_h, state_c = self.lstm1(net, initial_state=hidden)
        net, state_h, state_c = self.lstm1(net)
        hidden = [state_h, state_c]

        logits = self.actor_dense1(net) 
        actor = self.softmax(logits)
        log_logits = tf.nn.log_softmax(logits)

        critic = self.critic_dense1(net)
        critic = self.critic_dense2(critic)
        critic = tf.reshape(critic, [-1])

        self.actor = actor
        self.logits = logits
        self.log_logits = log_logits
        self.critic = critic
        self.hidden_state = hidden

        return actor, logits, log_logits, critic, hidden

    def build_loss(self, old_log_logit, action, advantage, td_target):
        def _log(val):
            return tf.log(tf.clip_by_value(val, 1e-10, 10.0))

        with tf.name_scope('trainer'):
            # Entropy
            entropy = -tf.reduce_mean(self.actor * _log(self.actor), name='entropy')

            # Critic Loss
            td_error = td_target - self.critic
            critic_loss = tf.reduce_mean(tf.square(td_error), name='critic_loss')

            # Actor Loss
            action_OH = tf.one_hot(action, self.action_size, dtype=tf.float32)
            log_prob = tf.reduce_sum(self.log_logits * action_OH, 1)
            old_log_prob = tf.reduce_sum(old_log_logit * action_OH, 1)

            # Clipped surrogate function (PPO)
            ratio = tf.exp(log_prob - old_log_prob)
            surrogate = ratio * advantage
            clipped_surrogate = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * advantage
            surrogate_loss = tf.minimum(surrogate, clipped_surrogate, name='surrogate_loss')
            actor_loss = -tf.reduce_mean(surrogate_loss, name='actor_loss')

            total_loss = actor_loss
            if self.entropy_beta != 0:
                total_loss = actor_loss + entropy * self.entropy_beta
            if self.critic_beta != 0:
                total_loss = actor_loss + critic_loss * self.critic_beta

            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy

        return total_loss

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
                self.prev_action_ = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32, name='preaction_hold')
                self.prev_reward_ = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32, name='prereward_hold')
                self.hidden_ = [tf.placeholder(shape=[None, 256], dtype=tf.float32, name='hiddenh_hold'),
                                tf.placeholder(shape=[None, 256], dtype=tf.float32, name='hiddenc_hold')]

                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                self.td_target_ = tf.placeholder(shape=[None], dtype=tf.float32, name='td_target_hold')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')

                # Build Network
                model = PPO_LSTM_V1(action_size);  self.model = model
                state_tuples = (self.state_input, self.prev_action_, self.prev_reward_, self.hidden_)
                self.actor, self.logits, self.log_logits, self.critic, self.hidden = model(state_tuples)
                loss = model.build_loss(self.old_logits_, self.action_, self.advantage_, self.td_target_)
                model.summary()

                # Build Trainer
                optimizer = tf.keras.optimizers.Adam(lr)
                self.gradients = optimizer.get_gradients(loss, model.trainable_variables)
                self.update_ops = optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))

    def hidden_init(self, batch_size):
        return [np.zeros([batch_size, 256]), np.zeros([batch_size, 256])]

    def run_network(self, states, return_action=True):
        feed_dict = {
                self.state_input: states[0],
                self.prev_action_: states[1],
                self.prev_reward_: states[2],
                self.hidden_[0] : states[3][0],
                self.hidden_[1] : states[3][1],
            }
        a_probs, critics, logits, hidden_h, hidden_c = self.sess.run([self.actor, self.critic, self.log_logits, self.hidden[0], self.hidden[1]], feed_dict)
        if return_action:
            actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
            return actions, critics, logits, hidden_h, hidden_c
        else:
            return a_probs, critics, logits, hidden_h, hidden_c

    def update_network(self, states, action, td_target, advantage, old_logit, prev_actions, prev_rewards, hhs, hcs, global_episodes, writer=None, log=False):
        feed_dict = {
                self.state_input: states,
                self.prev_action_: prev_actions,
                self.prev_reward_: prev_rewards,
                self.hidden_[0] : hhs,
                self.hidden_[1] : hcs,
                self.action_: action,
                self.td_target_: td_target,
                self.advantage_: advantage,
                self.old_logits_: old_logit
            }
        self.sess.run(self.update_ops, feed_dict)

        if log:
            ops = [self.model.actor_loss, self.model.critic_loss, self.model.entropy]
            aloss, closs, entropy = self.sess.run(ops, feed_dict)

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
        


class DQN:
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
