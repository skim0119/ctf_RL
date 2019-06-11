import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np


class REINFORCE:
    """ Build the graph for A3C model

    It includes minor features that helps to interact with the network

    Note:
        Placeholder is indicated by underscore '_' at the end of the variable name

    Features:
        gradient batch:
            If true, it provide 'accumulate' and 'clear' batch method.
            Instead of updating the network every episode, it accumulate the gradient,
                and update gradient descent with the accumulated gradient.
            It tends to provide more 'averaged' trajectory and gradient.
            Be aware, the accumulation will yield higher gradient value.
                Learning rate might need to be adjusted.

        include_lstm:
            If true, include lstm network inbetween convolution and fc network.
    """

    def __init__(self,
                 in_size,
                 action_size,
                 learning_rate,
                 grad_clip_norm=None,
                 entropy_beta=0.25,
                 scope='main',
                 sess=None,
                 gradient_batch=False,
                 ):
        self.in_size = [None]+in_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.entropy_beta = entropy_beta
        self.scope = scope
        self.sess = sess

        # Configurations
        self._gradient_batch = gradient_batch
        if self._gradient_batch:
            raise NotImplementedError

        # Input/Output Tag
        self.input_tag = 'Forward_input/state'
        self.output_tag = 'FC_layers/action'

        # LSTM Parameters
        self.gru_unit_size = 128
        self.gru_num_layers = 1

        # with tf.name_scope('self.scope'):
        self._build_placeholders()
        self._build_network()
        self._build_loss()
        self._build_optimizer()

    def _build_placeholders(self):
        """ Define the placeholders for forward and back propagation """
        with tf.name_scope('Forward_input'):
            self.state_input_ = tf.placeholder(shape=self.in_size, dtype=tf.float32, name='state')
            self.rnn_init_states = tuple(tf.placeholder(tf.float32, (None, self.gru_unit_size), name="rnn_init_states" + str(i))
                                         for i in range(self.gru_num_layers))
            self.seq_len = tf.placeholder(tf.int32, (None,), name="seq_len")

        with tf.name_scope('Backward_input'):
            self.action_ = tf.placeholder(shape=[None, None], dtype=tf.int32, name='action')
            self.reward_ = tf.placeholder(shape=[None, None], dtype=tf.float32, name='reward')
            self.actions_flatten = tf.reshape(self.action_, (-1,))
            self.actions_OH = tf.one_hot(self.actions_flatten, self.action_size)
            self.rewards_flatten = tf.reshape(self.reward_, (-1,))

    def _build_network(self):
        """ Define network """
        with tf.variable_scope('Conv_layers'):
            bulk_shape = tf.stack([tf.shape(self.state_input_)[0],
                                   tf.shape(self.state_input_)[1], 128])
            net = tf.reshape(self.state_input_, [-1]+self.in_size[2:])
            net = layers.conv2d(net, 16, [5, 5],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2, 2])
            net = layers.conv2d(net, 32, [3, 3],
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
            net = layers.fully_connected(net, 128, activation_fn=tf.nn.relu)
            net = tf.reshape(net, bulk_shape)

        with tf.variable_scope("RNN_layers"):
            gru_cell = tf.contrib.rnn.GRUCell(self.gru_unit_size)
            gru_cells = tf.contrib.rnn.MultiRNNCell([gru_cell] * self.gru_num_layers)
            net, self.final_state = tf.nn.dynamic_rnn(gru_cells,
                                                      net,
                                                      initial_state=self.rnn_init_states,
                                                      sequence_length=self.seq_len)
            net = tf.reshape(net, [-1, self.gru_unit_size])

        with tf.variable_scope('FC_layers'):
            # Fully Connected Layer
            self.logit = layers.fully_connected(net, self.action_size,
                                                activation_fn=None,
                                                scope='logit')
            self.output = tf.nn.softmax(self.logit, name='action')

    def _build_loss(self):
        """ Define loss """
        with tf.name_scope('Loss'):
            with tf.name_scope("masker"):
                num_step = tf.shape(self.state_input_)[1]
                self.mask = tf.sequence_mask(self.seq_len, num_step)
                self.mask = tf.reshape(tf.cast(self.mask, tf.float32), (-1,))

            self.entropy = -tf.reduce_sum(self.output * tf.log(self.output+1e-8), axis=1)
            self.entropy = tf.multiply(self.entropy, self.mask)
            self.entropy = tf.reduce_mean(self.entropy, name='entropy')

            self.loss = tf.reduce_mean(tf.square(self.output - self.actions_OH), axis=1)  # L2 loss
            self.loss = tf.multiply(self.loss, self.mask)
            self.loss = tf.reduce_mean(tf.multiply(self.loss, self.rewards_flatten))

            #obj_func = tf.log(tf.reduce_sum(self.output * self.actions_flatten, 1))
            #exp_v = obj_func * self.rewards_flatten * self.mask
            #self.loss = tf.reduce_mean(-exp_v, name='actor_loss')
            # self.loss = self.loss  # + self.entropy_beta * self.entropy

    def _build_optimizer(self):
        """ Define optimizer and gradient """
        with tf.name_scope('Trainer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.grads = self.optimizer.compute_gradients(self.loss)

            if self.grad_clip_norm is not None:
                self.grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in self.grads]

            if self._gradient_batch:
                grad_holders = [(tf.Variable(var, trainable=False, dtype=tf.float32, name=var.op.name+'_buffer'), var)
                                for var in tf.trainable_variables()]
                self.accumulate_gradient = tf.group(
                    [tf.assign_add(a[0], b[0]) for a, b in zip(grad_holders, self.grads)])
                self.clear_batch = tf.group([tf.assign(a[0], a[0]*0.0) for a in grad_holders])
                self.update_batch = self.optimizer.apply_gradients(grad_holders)
            else:
                self.update_batch = self.optimizer.apply_gradients(self.grads)

            # Debug Parameters
            self.grad_norm = tf.global_norm([grad for grad, var in self.grads])
            self.var_norm = tf.global_norm(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def _build_summary(self):
        raise NotImplementedError
        # Summary
        # Histogram output
        # with tf.variable_scope('debug_parameters'):
        #     tf.summary.histogram('output', self.output)
        #     tf.summary.histogram('action', self.action_holder)
        # # Graph summary Loss
        # with tf.variable_scope('summary'):
        #     tf.summary.scalar(name='total_loss', tensor=self.loss)
        #     tf.summary.scalar(name='Entropy', tensor=self.entropy)
        # with tf.variable_scope('weights_bias'):
        #     # Histogram weights and bias
        #     for var in slim.get_model_variables():
        #         tf.summary.histogram(var.op.name, var)
        # with tf.variable_scope('gradients'):
        #     # Histogram Gradients
        #     for var, grad in zip(slim.get_model_variables(), self.grads):
        #         tf.summary.histogram(var.op.name+'/grad', grad[0])

    def get_action(self, states, rnn_init_states, seq_len=[1]):
        # Only single action
        feed_dict = {self.state_input_: states,
                     self.rnn_init_states: rnn_init_states,
                     self.seq_len: seq_len}
        a_probs, final_state = self.sess.run([self.output, self.final_state], feed_dict=feed_dict)
        return np.random.choice(self.action_size, p=a_probs[0]), final_state

    # def gradient_clear(self):
    #    assert self._gradient_batch
    #    self.sess.run(self.clear_batch)

    # def gradient_accumulate(self, states, rewards, actions):
    #    assert self._gradient_batch
    #    feed_dict = {self.state_input_ : states,
    #                 self.rewards_ : rewards,
    #                 self.actions_ : actions}
    #    self.sess.run(self.accumulate_gradient, feed_dict=feed_dict)

    # def update_network_batch(self):
    #    assert self._gradient_batch
    #    self.sess.run(self.update_batch)

    def update_network(self, states, rewards, actions, rnn_init_states, seq_len):
        feed_dict = {self.state_input_: np.stack(states),
                     self.reward_: rewards,
                     self.action_: actions,
                     self.rnn_init_states[0]: np.stack(rnn_init_states),
                     self.seq_len: seq_len}
        self.sess.run(self.update_batch, feed_dict=feed_dict)

    def get_lstm_init(self):
        init_state = tuple([np.zeros((1, self.gru_unit_size)) for _ in range(self.gru_num_layers)])
        return init_state
