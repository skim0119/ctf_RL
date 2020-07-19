import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from math import sqrt

from utility.utils import store_args

from itertools import compress

""" Basic template for building new network module.

Notes:
    Placeholder is indicated by underscore '_' at the end of the variable name
"""


def initialize_uninitialized_vars(sess, global_vars=None):
    """
    Initialize uninitialized variables

    Parameters
    ----------------
    sess : [tensorflow.Session()] 
    """
    if global_vars is None:
        global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))
    non_init_length = len(not_initialized_vars)

    print(f'{non_init_length} number of non-initialized variables found.')

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        print('Initialized all non-initialized variables')

def put_ctf_state_on_grid(images, pad=1):
    # image : [num_image, y, x, num_channel]
    # Regularize
    i_min = tf.reduce_min(tf.reduce_min(images, axis=1, keepdims=True), axis=2, keepdims=True)
    i_max = tf.reduce_max(tf.reduce_max(images, axis=1, keepdims=True), axis=2, keepdims=True)
    images = (images - i_min) / (i_max - i_min)
    
    padding = tf.constant([[0,0], [pad,pad], [pad,pad], [0,0]])
    images = tf.pad(images, padding, mode='CONSTANT')

    images = tf.concat(tf.unstack(images, axis=0), axis=0) # [num_image*y, x, num_channel]
    images = tf.concat(tf.unstack(images, axis=-1), axis=-1) # [num_image*y, num_channel*x]

    # scale to [0, 255] and convert to uint8
    images = tf.image.convert_image_dtype(images, dtype = tf.uint8) 
    return tf.expand_dims(tf.expand_dims(images, -1), 0)
    

def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    https://gist.github.com/kukuruza/03731dc494603ceab0c5
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''


    kernel = tf.reduce_sum(kernel, axis=2, keepdims=True)
    if grid_Y == -1:
        grid_Y = kernel.get_shape()[3] // grid_X

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    padding = tf.constant([[pad, pad],[pad,pad], [0,0], [0,0]])
    x1 = tf.pad(kernel1, padding, mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    X = kernel1.get_shape()[0] + 2 * pad
    Y = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2)) # [kernel, Y, X, channel]
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_Y, Y * grid_X, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_Y, Y * grid_X, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8) 

def put_channels_on_grid (image, grid_Y, grid_X, pad = 1):
    '''
    Args:
      image:            tensor of shape [Y, X, NumChannels]
      (grid_Y, grid_X):  shape of the grid. Require: NumNumChannels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, 1].
    '''

    image = tf.expand_dims(image, 2)
    return put_kernels_on_grid (image, grid_Y, grid_X, pad = pad)

def put_flat_on_grid (image, grid_Y, grid_X, pad=1):
    image = tf.expand_dims(image, 1)
    image = tf.expand_dims(image, 1)
    return put_channels_on_grid (image, grid_Y, grid_X, pad = pad)

class Deep_layer:
    @staticmethod
    def conv2d_pool(input_layer, channels, kernels, pools=None, strides=None,
                    activation=tf.nn.relu, padding='SAME', flatten=False, reuse=False, return_summary=False):
        assert len(channels) == len(kernels)
        if strides is None:
            strides = [1] * len(channels)

        kernel_summary = []
        net = input_layer

        if return_summary:
            grid = put_channels_on_grid(net[0], -1, 8)
            kernel_summary.append(tf.summary.image('input_image', grid, max_outputs=1))

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

            # kernel summary
            if return_summary:
                with tf.variable_scope(f'conv_{idx}'):
                    tf.get_variable_scope().reuse_variables()
                    weights = tf.get_variable('weights')
                    grid = put_kernels_on_grid (weights, -1, 8)
                    kernel_summary.append(tf.summary.image(f'conv_{idx}/kernels', grid, max_outputs=1))
                grid = put_channels_on_grid (net[0], -1, 8)
                kernel_summary.append(tf.summary.image(f'conv_image_{idx}', grid, max_outputs=1))

        if flatten:
            net = layers.flatten(net)

        if return_summary:
            return net, tf.summary.merge(kernel_summary)

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
