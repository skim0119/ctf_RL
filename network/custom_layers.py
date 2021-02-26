import tensorflow as tf
import tensorflow.contrib.layers as layers

def grouped_conv(input_layer, n_group, axis=-1, **kwargs):
    with tf.variable_scope("GroupCNN"):
        input_groups = tf.split(input_layer, n_group, axis=axis)
        output_group = [
                layers.convolution(inputs=layer, **kwargs)
                for layer in input_groups
            ]
        output = tf.concat(output_group, axis=axis)
    return output

def grouped_sep_conv(input_layer, n_group, axis=-1, **kwargs):
    with tf.variable_scope("GroupSepCNN"):
        input_groups = tf.split(input_layer, n_group, axis=axis)
        output_group = [
                layers.separable_conv2d(inputs=layer, **kwargs)
                for layer in input_groups
            ]
        output = tf.concat(output_group, axis=axis)
    return output
