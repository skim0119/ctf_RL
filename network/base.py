import tensorflow as tf
#import tensorflow.keras.layers as layers
import tensorflow.contrib.layers as layers
import numpy as np

from utility.utils import store_args


""" Basic template for building new network module.

Notes:
    Placeholder is indicated by underscore '_' at the end of the variable name
"""


def initialize_uninitialized_vars(sess):
    """
    Initialize uninitialized variables

    Parameters
    ----------------
    sess : [tensorflow.Session()] 
    """
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))
    non_init_length = len(not_initialized_vars)

    print(f'{non_init_length} number of non-initialized variables found.')

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        print('Initialized all non-initialized variables')


class Deep_layer:
    @staticmethod
    def conv2d_pool(input_layer, channels, kernels, pools=None, strides=None,
                    activation=tf.nn.relu, padding='SAME', flatten=False, reuse=False):
        assert len(channels) == len(kernels)
        if strides is None:
            strides = [1] * len(channels)
        net = input_layer
        for idx, (ch, kern, pool, stride) in enumerate(zip(channels, kernels, pools, strides)):
            net = layers.conv2d(
                net, ch, kern, stride,
                activation_fn=activation,
                padding=padding,
                weights_initializer=layers.xavier_initializer_conv2d(),
                biases_initializer=tf.zeros_initializer(),
                reuse=reuse,
                scope=f'conv_{idx}')
            if pools is not None and pools[idx] > 1:
                net = layers.max_pool2d(net, pool)
        if flatten:
            net = layers.flatten(net)
        return net

    @staticmethod
    def fc(input_layer, hidden_layers, dropout=1.0,
           activation=tf.nn.elu, reuse=False, scope=""):
        net = input_layer
        # init = Custom_initializers.variance_scaling()
        for idx, node in enumerate(hidden_layers):
            net = layers.fully_connected(net, int(node),
                                         activation_fn=activation,
                                         scope=f"dense_{idx}" + scope,
                                         # weights_initializer=init,
                                         reuse=reuse)
            if idx < len(hidden_layers) - 1:
                net = layers.dropout(net, dropout)
        return net


class Tensor_logger:
    @store_args
    def __init__(self, log_path, summary_name, sess, writer_id='0'):
        self.writer = tf.summary.FileWriter(log_path, filename_suffix=writer_id)
        #self.scalar_logger = Tensorboard_utility.scalar_logger

    def log_scalar(self, tag, value, step):
        #with tf.summary.FileWriter(self.log_path) as writer:
        #    tag = self.summary_name + '/' + tag
        #    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #    writer.add_summary(summary, step)
        #    writer.flush()

        tag = self.summary_name + '/' + tag
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        #self.writer.flush()
        #self.scalar_logger(self.summary_name + '/' + tag, value, step, self.writer)

        

    def set_histograms(self, var_list):
        '''for var in tf.trainable_variables(scope=global_scope):
            tf.summary.histogram(var.name, var)
        merged_summary_op = tf.summary.merge_all()'''
        pass


class Tensorboard_utility:
    @staticmethod
    def scalar_logger(tag, value, step, writer):
        """
        Log a single scalar variable.

        Parameter
        ----------
        tag : [string]
            Name of the scalar (name of the plot)
        value : [float]
            value to record
        step : [int]
            training iteration
        writer : [tf.summary.FileWriter]
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)

    @staticmethod
    def histogram_logger(summary, step, writer):
        writer.add_summary(summary, step)

    @staticmethod
    def variable_statistic_logger(
        var, include_min=False, include_max=False, mean=True,
        std=False, histogram=False, name_scope='summaries'
    ):
        """
        Log variable statistic for a Tensorboard visualization

        Parameters
        ----------------

        var : [Tensor]
             Scalar variable

        include_min, include_max, mean, std : [bool]
            Toggle which statistic to include

        histogram : [bool]
            Toggle to include histogram

        name_scope : [string]

        Returns
        ----------------

        list : [List]
            list of the summary Tensor

        """
        summaries = []
        with tf.name_scope(name_scope):
            if mean:
                summaries.append(tf.summary.scalar('mean', tf.reduce_mean(var)))
            if std:
                summaries.append(tf.summary.scalar('stddev',
                    tf.sqrt(tf.reduce_mean(tf.square(var - mean)))))
            if include_max:
                summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
            if include_min:
                summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
            if histogram:
                summaries.append(tf.summary.histogram('histogram', var))
        return summaries


class Custom_initializers:
    @staticmethod
    def normalized_columns_initializer(std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    @staticmethod
    def log_uniform_initializer(mu, std):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.lognormal(mean=mu, sigma=std, size=shape).astype(np.float32)
            return tf.constant(out)
        return _initializer

    @staticmethod
    def variance_scaling():
        return tf.contrib.layers.variance_scaling_initializer(factor = 1.0, mode = "FAN_AVG", uniform = False)
