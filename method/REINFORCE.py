import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np
import random

import utility

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
        self.in_size = in_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.entropy_beta=entropy_beta
        self.scope = scope
        self.sess = sess

        ## Configurations
        self._gradient_batch = gradient_batch

        #with tf.name_scope('self.scope'):
        self._build_placeholders()
        self._build_network()
        self._build_loss()
        self._build_optimizer()

        ## Input/Output Tag
        self.input_tag = 'Forward_input/state'
        self.output_tag = 'FC_layers/action'

    def _build_placeholders(self):
        """ Define the placeholders for forward and back propagation """
        with tf.name_scope('Forward_input'):
            self.state_input_ = tf.placeholder(shape=self.in_size,dtype=tf.float32, name='state')

        with tf.name_scope('Backward_input'):
            self.action_ = tf.placeholder(shape=[None],dtype=tf.int32, name='action')
            self.action_OH = tf.one_hot(self.action_holder, self.action_size)
            self.reward_ = tf.placeholder(shape=[None],dtype=tf.float32, name='reward')

    def _build_network(self):
        """ Define network """
        with tf.variable_scope('Conv_layers'):
            net = layers.conv2d(self.state_input, 16, [5,5],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2,2])
            net = layers.conv2d(net, 32, [3,3],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.max_pool2d(net, [2,2])
            net = layers.conv2d(net, 32, [2,2],
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer_conv2d(),
                                biases_initializer=tf.zeros_initializer(),
                                padding='SAME')
            net = layers.flatten(net)

        with tf.variable_scope('FC_layers'):
            ## Fully Connected Layer
            net = layers.fully_connected(net, 128,
                                         activation_fn=tf.nn.relu)
            self.output = layers.fully_connected(net, self.action_size,
                                                 activation_fn=tf.nn.softmax,
                                                 scope='action')

    def _build_loss(self):
        """ Define loss """
        with tf.name_scope('Loss'):
            # Update Operations
            self.entropy = -tf.reduce_mean(self.output * tf.log(self.output+1e-8), name='entropy') # measure action diversity
            obj_func = tf.log(tf.reduce_sum(self.output * self.action_OH, 1))
            exp_r = obj_func * self.reward_ + self.entropy_beta * self.entropy
            self.loss = tf.reduce_mean(-exp_r, name='loss')

    def _build_optimizer(self):
        """ Define optimizer and gradient """
        with tf.name_scope('Trainer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.grads = self.optimizer.compute_gradients(self.loss)
            if self.grad_clip_norm is not None:
                self.grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in self.grads]

            if self._gradient_batch:
                grad_holders = [(tf.Variable(var, trainable=False, dtype=tf.float32, name=var.op.name+'_buffer'), var) for var in tf.trainable_variables()]
                self.accumulate_gradient = tf.group([tf.assign_add(a[0],b[0]) for a,b in zip(grad_holders, self.grads)]) 
                self.clear_batch = tf.group([tf.assign(a[0],a[0]*0.0) for a in grad_holders])
                self.update_batch = self.optimizer.apply_gradients(grad_holders) 
            else:
                self.update_batch = self.optimizer.apply_gradients(self.grads)

    def _build_summary(self):
        raise NotImplementedError
        # Summary
        # Histogram output
        with tf.variable_scope('debug_parameters'):
            tf.summary.histogram('output', self.output)   
            tf.summary.histogram('action', self.action_holder)
        
        # Graph summary Loss
        with tf.variable_scope('summary'):
            tf.summary.scalar(name='total_loss', tensor=self.loss)
            tf.summary.scalar(name='Entropy', tensor=self.entropy)
        
        with tf.variable_scope('weights_bias'):
            # Histogram weights and bias
            for var in slim.get_model_variables():
                tf.summary.histogram(var.op.name, var)
                
        with tf.variable_scope('gradients'):
            # Histogram Gradients
            for var, grad in zip(slim.get_model_variables(), self.grads):
                tf.summary.histogram(var.op.name+'/grad', grad[0])

    def get_action(self, states, deterministic=False):
        a_probs = self.sess.run(self.output, feed_dict={self.state_input_ : states})
        if deterministic:
            return np.argmax(a_probs, axis=1)
        else:
            return [np.random.choice(self.action_size, p=prob/sum(prob)) for prob in a_probs]

    def gradient_clear(self):
        assert self._gradient_batch
        self.sess.run(self.clear_batch)

    def gradient_accumulate(self, states, rewards, actions):
        assert self._gradient_batch
        feed_dict = {self.state_input_ : states,
                     self.rewards_ : rewards,
                     self.actions_ : actions}
        self.sess.run(self.accumulate_gradient, feed_dict=feed_dict)

    def update_network_batch(self):
        assert self._gradient_batch
        self.sess.run(self.update_bath)

    def update_network(self, states, rewards, actions):
        assert not self._gradient_batch
        feed_dict = {self.state_input_ : states,
                     self.rewards_ : rewards,
                     self.actions_ : actions}
        self.sess.run(self.update_batch, feed_dict=feed_dict)
