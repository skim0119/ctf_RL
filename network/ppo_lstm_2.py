""" Proximal Policy Optimization (PPO) + LSTM layer """
import tensorflow as tf
import tensorflow.layers as layers

import numpy as np
import time

from utility.dataModule import one_hot_encoder_v2 as one_hot_encoder
from utility.utils import discount_rewards

import tqdm

# Network configuration
SERIAL_SIZE = 64
RNN_UNIT_SIZE = 64
RNN_NUM_LAYERS = 2

MINIBATCH_SIZE = 1 
SEQUENCE_SIZE = 2
L2_REGULARIZATION = 0.001


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class PPO:
    """ Proximal Policy Optimization with LSTM (TensorFlow)

    Module contains network structure and pipeline for Actor-Critic.
    Pipeline structure support asynchronous training.
    Network includes CNN, FC, and RNN (GRU) layers.
    """

    def __init__(self,
                 in_size,
                 action_size,
                 scope,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 grad_clip_norm=0,
                 global_step=None,
                 critic_beta=1.0,
                 entropy_beta=0.01,
                 sess=None,
                 global_network=None,
                 separate_train=False,
                 rnn_type='GRU'
                 ):
        # Class Environment
        self.sess = sess

        # Parameters & Configs
        self.in_size = in_size
        self.action_size = action_size
        self.scope = scope
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.grad_clip_norm = grad_clip_norm
        self.global_step = global_step
        self.critic_beta = critic_beta
        self.entropy_beta = entropy_beta
        self.global_network = global_network
        self.separate_train = separate_train

        # Input/Output
        self.input_tag = self.scope + '/Forward_input/state'
        self.output_tag = self.scope + '/actor/action'

        # RNN Configurations
        self.rnn_type = rnn_type
        
        if scope != 'global':
            self.param_summary = None
            self.loss_summary = None
            self.grad_summary = None

        with tf.variable_scope(self.scope):
            self._build_placeholders()
            self._build_dataset()
            # Target Network
            self.policy_target, self.policy_target_params, _, _ = self._build_actor_network(
                self.batch['state'], 'actor_target', batch_size=MINIBATCH_SIZE)
            self.value_target, self.value_target_params, _, _ = self._build_critic_network(
                self.batch['state'], 'critic_target', batch_size=MINIBATCH_SIZE)
            # Policy Network for Backpropagation
            self.actions, self.actions_params, self.action_init_state, self.action_fin_state = self._build_actor_network(
                self.batch["state"], 'actor', batch_size=MINIBATCH_SIZE)
            self.critics, self.critics_params, self.critic_init_state, self.critic_fin_state = self._build_critic_network(
                self.batch["state"], 'critic', batch_size=MINIBATCH_SIZE)
            # Policy Network for Action Generation (batch_size=1)
            self.action_eval, _, self.action_eval_init_state, self.action_eval_fin_state = self._build_actor_network(self.state_, 'actor', reuse=True)
            self.critic_eval, _, self.critic_eval_init_state, self.critic_eval_fin_state = self._build_critic_network(
                self.state_, 'critic', reuse=True)
            self.stoch_action = tf.squeeze(self.action_eval.sample(1), axis=0, name="sample_action")
            self.determ_action = self.action_eval.mode()
            self.params = self.actions_params + self.critics_params
            self.target_params = self.policy_target_params + self.value_target_params
            self._build_losses()
            self._build_pipeline()
            
            variable_summary = []
            for var in self.params:
                var_name = var.name+'_var'
                var_name = var_name.replace(':','_')
                variable_summary.append(tf.summary.histogram(var_name, var)) 
            self.param_summary = tf.summary.merge(variable_summary)


    def _build_dataset(self):
        # Use the TensorFlow Dataset API
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state_,
                                                           "actions": self.actions_,
                                                           "rewards": self.rewards_,
                                                           "advantage": self.advantages_})
        self.dataset = self.dataset.batch(SEQUENCE_SIZE, drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        self.batch = self.iterator.get_next()

    def _build_placeholders(self):
        """ Define the placeholders for forward and back propagation """
        self.state_ = tf.placeholder(tf.float32, self.in_size, 'state')
        self.actions_ = tf.placeholder(tf.int32, [None, 1], 'action')

        self.advantages_ = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards_ = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.keep_prob_ = tf.placeholder_with_default(tf.constant(1.0), shape=None)

        self.likelihood_ = tf.placeholder(shape=[None], dtype=tf.float32, name='likelihood_holder')
        self.likelihood_cumprod_ = tf.placeholder(shape=[None], dtype=tf.float32, name='likelihood_cumprod_holder')

    def _build_actor_network(self, state_in, name, reuse=False, batch_size=1):
        w_reg = None

        with tf.variable_scope(name, reuse=reuse):
            serial_net = layers.flatten(state_in)
            serial_net = layers.dense(serial_net, 256)
            serial_net = layers.dense(serial_net, RNN_UNIT_SIZE)

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=RNN_UNIT_SIZE, name='lstm_cell')
            #lstm = tf.nn.rnn_cell.ResidualWrapper(lstm)
            #lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob_)
            lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * RNN_NUM_LAYERS)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(serial_net, axis=0)

            rnn_net, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            rnn_net = tf.reshape(rnn_net, [-1, RNN_UNIT_SIZE], name='flatten_rnn_outputs')

            logits = layers.dense(serial_net, 5, 
                                  kernel_initializer=normalized_columns_initializer(0.01),
                                  kernel_regularizer=w_reg, name="pi_logits")
            policy_dist = tf.distributions.Categorical(logits=logits)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + name)
        return policy_dist, params, init_state, final_state

    def _build_critic_network(self, state_in, name, reuse=False, batch_size=1):
        w_reg = None# tf.contrib.layers.l2_regularizer(L2_REGULARIZATION)

        with tf.variable_scope(name, reuse=reuse):
            serial_net = layers.flatten(state_in)
            serial_net = layers.dense(serial_net, 256)
            serial_net = layers.dense(serial_net, RNN_UNIT_SIZE)

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=RNN_UNIT_SIZE, name='lstm_cell')
            #lstm = tf.nn.rnn_cell.ResidualWrapper(lstm)
            #lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob_)
            lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * RNN_NUM_LAYERS)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(serial_net, axis=0)

            rnn_net, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            rnn_net = tf.reshape(rnn_net, [-1, RNN_UNIT_SIZE], name='flatten_rnn_outputs')

            critic = layers.dense(serial_net, 1, 
                                  kernel_initializer=normalized_columns_initializer(1.0),
                                  kernel_regularizer=w_reg, name="critic_out")

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + name)
        return critic, params, init_state, final_state

    def _build_losses(self):
        epsilon = 0.3
        with tf.variable_scope('loss'):
            self.global_step = tf.train.get_or_create_global_step()
            epsilon_decay = tf.train.polynomial_decay(epsilon, self.global_step, 1e6, 0.01, power=1.0)

            with tf.variable_scope('actor'):
                ratio = tf.maximum(self.actions.prob(self.batch["actions"]), 1e-13) / \
                    tf.maximum(self.policy_target.prob(self.batch["actions"]), 1e-13)
                ratio = tf.clip_by_value(ratio, 0, 10)
                ppo1 = self.batch["advantage"] * ratio
                ppo2 = self.batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                self.actor_loss = -tf.reduce_mean(tf.minimum(ppo1, ppo2))

            with tf.variable_scope('critic'):
                clipped_value_estimate = self.value_target + tf.clip_by_value(self.critics - self.value_target, -epsilon_decay, epsilon_decay)
                critic_v1 = tf.squared_difference(clipped_value_estimate, self.batch["rewards"])
                critic_v2 = tf.squared_difference(self.critics, self.batch["rewards"])
                self.critic_loss = tf.reduce_mean(tf.maximum(critic_v1, critic_v2))
                # critic_loss= tf.reduce_mean(tf.square(self.critics - self.batch["rewards"]))

            with tf.variable_scope('entropy'):
                self.entropy = tf.reduce_mean(self.actions.entropy())

            self.total_loss = self.actor_loss + (self.critic_loss * self.critic_beta) - self.entropy * self.entropy_beta
            
            summaries = []
            summaries.append(tf.summary.scalar("epsilon", epsilon_decay))
            summaries.append(tf.summary.scalar("total_loss", self.total_loss))
            summaries.append(tf.summary.scalar("actor_loss", self.actor_loss))
            summaries.append(tf.summary.scalar("critic_loss", self.critic_loss))
            summaries.append(tf.summary.scalar("entropy", self.entropy))
            self.loss_summary = tf.summary.merge(summaries)

    def _build_pipeline(self):
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr_actor)
            self.grads = optimizer.compute_gradients(self.total_loss, var_list=self.params)
            self.train_op = optimizer.minimize(self.total_loss,
                                               global_step=self.global_step,
                                               var_list=self.params)

        with tf.variable_scope('push'):
            update_target_op_actor = [target.assign(p) for p, target in zip(self.actions_params, self.policy_target_params)]
            update_target_op_critic = [target.assign(p) for p, target in zip(self.critics_params, self.value_target_params)]
            self.update_target_op = tf.group(update_target_op_actor, update_target_op_critic)

        summaries = []
        for grad, var in self.grads: 
            if grad is None:
                continue
            var_name = var.name + '_grad'
            var_name = var_name.replace(':', '_')
            summaries.append(tf.summary.histogram(var_name, grad))
        self.grad_summary = tf.summary.merge(summaries)

    def feed_forward(self, state, rnn_state):
        eval_ops = [self.stoch_action, self.critic_eval, self.action_eval_fin_state, self.critic_eval_fin_state]
        feed_dict = {self.state_: state,
                     self.action_eval_init_state: rnn_state[0],
                     self.critic_eval_init_state: rnn_state[1],
                     self.keep_prob_: 1.0}

        action, value, a_fin_state, c_fin_state = self.sess.run(eval_ops, feed_dict)
        return action[0], value[0], (a_fin_state, c_fin_state)

    def feed_backward(self, episode_rollouts, epochs=1):
        self.sess.run([self.update_target_op])

        c = 0
        ctime = time.time()
        for ep in range(epochs):
            np.random.shuffle(episode_rollouts)
            for ep_s, ep_a, ep_r, ep_adv in episode_rollouts:
                feed_dict = {self.state_: ep_s,
                             self.actions_: ep_a,
                             self.rewards_: ep_r,
                             self.advantages_: ep_adv}
                self.sess.run(self.iterator.initializer, feed_dict=feed_dict)

                a_state, c_state = self.sess.run([self.action_init_state, self.critic_init_state])
                summary_ = tf.summary.merge([self.param_summary, self.loss_summary, self.grad_summary])
                train_ops = [summary_, self.global_step, self.action_fin_state, self.critic_fin_state, self.train_op]

                while True:  # run until batch run out
                    try:
                        feed_dict = {self.action_init_state: a_state,
                                     self.critic_init_state: c_state,
                                     self.keep_prob_: 1.0}
                        summary, step, a_state, c_state, _ = self.sess.run(train_ops, feed_dict=feed_dict)
                        c += 1
                    except tf.errors.OutOfRangeError:
                        break
        dur = time.time() - ctime
        # print(f'Train: {c} batches trained, {len(episode_rollouts)} episodes: {dur} sec')
        return summary


if __name__ == '__main__':
    import gym
