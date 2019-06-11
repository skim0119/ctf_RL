""" Recurent Policy Gradient """
import tensorflow as tf
import tensorflow.layers as layers

import numpy as np

# Network configuration
RNN_UNIT_SIZE = 64
RNN_NUM_LAYERS = 1

MINIBATCH_SIZE = 8
L2_REGULARIZATION = 0.001


class Custom_initializer:
    @staticmethod
    def normalized_columns_init(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer


class RPG:
    """ Recurrent Policy Gradient with LSTM (TensorFlow)

    Module contains network structure and pipeline for dataset.
    Network includes LSTM and FC

    Implementation based on paper:
        http://people.idsia.ch/~juergen/joa2009.pdf

    Code is adapted to the environment CtF.
    """

    def __init__(self,
                 in_size,
                 action_size,
                 lr_policy=1e-3,
                 lr_baseline=1e-3,
                 entropy_beta=0.01,
                 tau=0.1,
                 sess=None,
                 ):
        # Class Environment
        self.sess = sess

        # Parameters & Configs
        self.in_size = in_size
        self.action_size = action_size
        self.lr_policy = lr_policy
        self.lr_baseline = lr_baseline
        self.entropy_beta = entropy_beta
        self.tau = tau

        # Input/Output
        # self.input_tag = self.scope + '/Forward_input/state'
        # self.output_tag = self.scope + '/actor/action'

        # Build Graph
        self._build_placeholders()
        self._build_dataset()
        # Target Network
        policy_targ, policy_targ_vars, _, _ = self._build_policy_network(
            self.batch["observations"], 'policy_target')
        baseline_targ, baseline_targ_vars, _, _ = self._build_baseline_network(
            self.batch["observations"], 'baseline_target')
        self.target_result = [policy_targ, baseline_targ]
        self.target_var = policy_targ_vars + baseline_targ_vars

        # Policy Network for Backpropagation
        policy, policy_vars, policy_init, policy_fin = self._build_policy_network(
            self.batch["observations"], 'policy')
        baseline, baseline_vars, baseline_init, baseline_fin = self._build_baseline_network(
            self.batch["observations"], 'baseline')
        self.result = [policy, baseline]
        self.policy_var = policy_vars
        self.baseline_var = baseline_vars
        self.graph_var = policy_vars + baseline_vars
        self.rnn_init = tuple([policy_init, baseline_init])
        self.rnn_fin = tuple([policy_fin, baseline_fin])

        # Policy Network for Forward Pass (batch_size=1)
        policy_eval, _, policy_eval_init, policy_eval_fin = self._build_policy_network(
            self.observation_, 'policy', reuse=True, batch_size=1)
        baseline_eval, _, baseline_eval_init, baseline_eval_fin = self._build_baseline_network(
            self.observation_, 'baseline', reuse=True, batch_size=1)
        self.evaluate = [policy_eval, baseline_eval]
        self.rnn_eval_init = tuple([policy_eval_init, baseline_eval_init])
        self.rnn_eval_fin = tuple([policy_eval_fin, baseline_eval_fin])

        with tf.variable_scope('push'):
            self.update_targ_op = [targ.assign(p * self.tau) for p, targ in zip(policy_vars, policy_targ_vars)] + [targ.assign(p * self.tau) for p, targ in zip(baseline_vars, baseline_targ_vars)]
            
        # Build Summary and Training Operations
        variable_summary = []
        for var in self.graph_var:
            var_name = var.name + '_var'
            var_name = var_name.replace(':', '_')
            variable_summary.append(tf.summary.histogram(var_name, var))
        self.var_summary = tf.summary.merge(variable_summary)
        self.loss_summary = self._build_losses()
        self.grad_summary = self._build_pipeline()

    def _build_placeholders(self):
        """ Define the placeholders """
        self.observation_ = tf.placeholder(tf.float32, [None] + self.in_size, 'observations')
        self.actions_ = tf.placeholder(tf.int32, [None, None], 'actions')
        self.actions_OH = tf.one_hot(self.actions_, self.action_size)
        self.rewards_ = tf.placeholder(tf.float32, [None, None], 'rewards')
        self.baseline_ = tf.placeholder(tf.float32, [None, None], 'baselines')
        self.seq_len_ = tf.placeholder(tf.float32, (None,), 'seq_len')

    def _build_dataset(self):
        """ Use the TensorFlow Dataset API """
        self.dataset = tf.data.Dataset.from_tensor_slices({"observations": self.observation_,
                                                           "actions": self.actions_,
                                                           "rewards": self.rewards_,
                                                           "baselines": self.baseline_})
        self.dataset = self.dataset.batch(MINIBATCH_SIZE, drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        self.batch = self.iterator.get_next()

    def _build_policy_network(self, state_in, name, reuse=False, batch_size=MINIBATCH_SIZE):
        w_reg = None

        with tf.variable_scope(name, reuse=reuse):
            state_in = layers.dense(state_in, RNN_UNIT_SIZE)
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=RNN_UNIT_SIZE, name='lstm_cell')
            #lstm = tf.nn.rnn_cell.ResidualWrapper(lstm)
            lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * RNN_NUM_LAYERS)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            rnn_net, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                                     inputs=state_in,
                                                     initial_state=init_state)
            rnn_net = tf.reshape(rnn_net, [-1, RNN_UNIT_SIZE])

            logits = layers.dense(rnn_net, self.action_size,
                                  kernel_initializer=Custom_initializer.normalized_columns_init(0.01),
                                  kernel_regularizer=w_reg,
                                  name='pi_logits')
            dist = tf.nn.softmax(logits, name='action')
            policy_dist = tf.reshape(dist, [batch_size, -1, self.action_size])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return policy_dist, vars, init_state, final_state

    def _build_baseline_network(self, state_in, name, reuse=False, batch_size=MINIBATCH_SIZE):
        w_reg = None #tf.contrib.layers.l2_regularizer(L2_REGULARIZATION)

        with tf.variable_scope(name, reuse=reuse):
            state_in = layers.dense(state_in, RNN_UNIT_SIZE)
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=RNN_UNIT_SIZE, name='lstm_cell')
            #lstm = tf.nn.rnn_cell.ResidualWrapper(lstm)
            lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * RNN_NUM_LAYERS)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

            rnn_net, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=state_in, initial_state=init_state)
            rnn_net = tf.reshape(rnn_net, [-1, RNN_UNIT_SIZE])

            critic = layers.dense(rnn_net, 1,
                                  kernel_initializer=Custom_initializer.normalized_columns_init(1.0),
                                  kernel_regularizer=w_reg, name="critic_out")
            critic = tf.reshape(critic, [batch_size, -1])

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return critic, vars, init_state, final_state

    def _build_losses(self):
        with tf.variable_scope('loss'):
            self.global_step = tf.train.get_or_create_global_step()
            
            mask = tf.sequence_mask(self.seq_len_, dtype=tf.float32)

            with tf.variable_scope('policy'):
                td = self.batch["rewards"] - self.batch["baselines"]  # (R-B(ht))
                obj_func = tf.log(tf.reduce_sum(self.result[0] * tf.one_hot(self.batch["actions"], self.action_size), 2))  # log_pi(a|h)
                exp_v = obj_func * td * mask
                #self.policy_loss = -tf.reduce_mean(exp_v)
                traj_loss = tf.reduce_sum(exp_v, axis=1)# / self.seq_len_
                self.policy_loss = -tf.reduce_sum(traj_loss, name='policy_loss')/tf.reduce_sum(mask)

            with tf.variable_scope('baseline'):
                err = self.target_result[1] - self.batch["rewards"]
                #self.baseline_loss = tf.reduce_mean(err)
                traj_err = tf.reduce_sum(tf.square(err)*mask, axis=1)# / self.seq_len_
                self.baseline_loss = tf.reduce_sum(traj_err)/tf.reduce_sum(mask)

            with tf.variable_scope('entropy'):
                self.entropy = -tf.reduce_mean(self.result[0] * tf.log(self.result[0]), name='entropy')

            summaries = []
            summaries.append(tf.summary.scalar("policy_loss", self.policy_loss))
            summaries.append(tf.summary.scalar("baseline_loss", self.baseline_loss))
            summaries.append(tf.summary.scalar("entropy", self.entropy))

            return tf.summary.merge(summaries)

    def _build_pipeline(self):
        with tf.variable_scope('train'):
            policy_optimizer = tf.train.AdamOptimizer(self.lr_policy)
            baseline_optimizer = tf.train.AdamOptimizer(self.lr_baseline)
            self.train_op = tf.group([policy_optimizer.minimize(self.policy_loss,
                                                                var_list=self.policy_var),
                                      baseline_optimizer.minimize(self.baseline_loss,
                                                                  global_step=self.global_step,
                                                                  var_list=self.baseline_var)])

        #with tf.variable_scope('push'):
        #    self.update_targ_op = [targ.assign(p * self.tau)
        #                           for p, targ in zip(self.graph_var, self.target_var)]

        summaries = []
        #for grad, var in self.grads:
        #    var_name = var.name + '_grad'
        #    var_name = var_name.replace(':', '_')
        #    summaries.append(tf.summary.histogram(var_name, grad))
        return None#tf.summary.merge(summaries)

    def feed_forward(self, state, rnn_state):
        eval_ops = self.evaluate + [self.rnn_eval_init, self.rnn_eval_fin]
        feed_dict = {self.observation_: state,
                     self.rnn_eval_init[0]: rnn_state[0],
                     self.rnn_eval_init[1]: rnn_state[1]}

        action_dist, value, p_fin_state, b_fin_state = self.sess.run(eval_ops, feed_dict)
        action_dist = np.squeeze(action_dist)
        action = np.random.choice(self.action_size, p=action_dist/ sum(action_dist))
        value = np.squeeze(value)

        return action, value, (p_fin_state[0], b_fin_state[0])

    def feed_backward(self, episode_rollouts, epochs=1):
        for ep in range(epochs):
            np.random.shuffle(episode_rollouts)
            lengths = [len(p[0]) for p in episode_rollouts]
            maxlen = max(lengths)

            observations, actions, rewards, baselines = [], [], [], []
            for ep_s, ep_a, ep_r, ep_base in episode_rollouts:
                length = len(ep_s)
                observations.append(np.append(ep_s, np.zeros((maxlen - length,) + ep_s.shape[1:]), axis=0))
                actions.append(np.append(ep_a, np.zeros(maxlen - length)))
                rewards.append(np.append(ep_r, np.zeros(maxlen - length)))
                baselines.append(np.append(ep_base, np.zeros(maxlen - length)))
                
            observations = np.stack(observations)
            actions = np.stack(actions)
            rewards = np.stack(rewards)
            baselines = np.stack(baselines)
            feed_dict = {self.observation_: observations,
                         self.actions_: actions,
                         self.rewards_: rewards,
                         self.baseline_: baselines}
            self.sess.run(self.iterator.initializer, feed_dict=feed_dict)

            p_state, b_state = self.sess.run(self.rnn_init)
            summary_ = tf.summary.merge([self.var_summary, self.loss_summary])# , self.grad_summary])
            train_ops = [summary_, self.global_step,
                         self.rnn_fin[0], self.rnn_fin[1], self.train_op]
            pull_input = [self.batch["actions"], self.batch["rewards"]]

            summary = None
            step = 0
            while True:  # run until batch run out
                try:
                    feed_dict = {self.rnn_init[0]: p_state,
                                 self.rnn_init[1]: b_state,
                                 self.seq_len_: np.array(lengths[:MINIBATCH_SIZE])}
                    lengths = lengths[MINIBATCH_SIZE:]
                    summary, step, p_state, b_state, _, action, reward = self.sess.run(train_ops+pull_input, feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break
        self.sess.run([self.update_targ_op])
        return summary, step


if __name__ == '__main__':
    pass
