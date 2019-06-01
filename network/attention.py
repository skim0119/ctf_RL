import tensorflow as tf

import numpy as np

def soft_attention(h_prev, a, num_input, hidden_size):
    def weight_variable(name, shape):
        return tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    # h_prev: output from lstm of previous time step (shape: [batch_size, lstm_size])
    # a: Result of CNN [batch_size, conv_size * conv_size, channel_size] 

    Wa = weight_variable('Wa', [num_input, 1])
    Wh = weight_variable('Wh', [hidden_size, 1])

    m_list = [tf.tanh(tf.matmul(a[i], Wa) + tf.matmul(h_prev, Wh)) for i in range(len(a))] 
    m_concat = tf.concat([m_list[i] for i in range(len(a))], axis = 1)     
    alpha = tf.nn.softmax(m_concat) 
    z_list = [tf.multiply(a[i], tf.slice(alpha, (0, i), (-1, 1))) for i in range(len(a))]
    z_stack = tf.stack(z_list, axis = 2)
    z = tf.reduce_sum(z_stack, axis = 2)

    return alpha, z

def self_attention(data, hidden_dim, output_dim, residual=True):
    # Motivated from 'Attention is all you need'
    # data shape : [T,C]
    # output_dim : V
    # output shape : [T, C+V]
    def scaled_dot_product(Q, K, scaled_=True, masked_=False):
        # Scaled-dot product
        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        if masked_:
            raise NotImplementedError

        attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
        return attention

    Q = tf.layers.dense(data, hidden_dim)  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, hidden_dim)  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, output_dim)  # [batch_size, sequence_length, output_dim]

    attention = scaled_dot_product(Q, K)  # [batch_size, sequence_length, sequence_length]
    output = tf.matmul(attention, V)  # [batch_size, sequence_length, output_dim]
    
    if residual:
        #output = data + output
        output = tf.concat([data, output], axis=1)

    return output

def non_local_nn_2d(data, hidden_dim, pool=False, name='non_local', summary_adder=None):
    # data shape : [Batch, H, W, Channel]
    # output dim : [Batch, H, W, Channel]
    with tf.variable_scope(name):
        def scaled_dot_product(Q, K, scaled_=True, masked_=False):
            # Scaled-dot product
            attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

            if scaled_:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

            if masked_:
                raise NotImplementedError

            attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
            return attention

        nbatch, h, w, output_dim = data.get_shape().as_list()
        Q = tf.contrib.layers.convolution(data, hidden_dim, 1)
        Q = tf.reshape(Q, [-1, h*w, hidden_dim])

        if pool:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.contrib.layers.max_pool2d(K, 2)
            K = tf.reshape(K, [-1, (h//2)*(w//2), hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.contrib.layers.max_pool2d(V, 2)
            V = tf.reshape(V, [-1, (h//2)*(w//2), hidden_dim])
        else:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.reshape(K, [-1, h*w, hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.reshape(V, [-1, h*w, hidden_dim])

        dot = scaled_dot_product(Q, K)
        output = tf.matmul(dot, V)  # [batch_size, sequence_length, output_dim]
        output = tf.reshape(output, [-1,h,w,hidden_dim])
        output = tf.contrib.layers.convolution(output, output_dim, 1)
        output = output + data  # Residual

        return output

def non_local_nn(data, hidden_dim, pool=False, name='non_local'):
    # data shape : [Batch, T, H, W, Channel]
    # output dim : [Batch, T, H, W, Channel]
    with tf.variable_scope(name):
        def scaled_dot_product(Q, K, scaled_=True, masked_=False):
            # Scaled-dot product
            attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

            if scaled_:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

            if masked_:
                raise NotImplementedError

            attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]
            return attention

        nbatch, t, h, w, output_dim = data.get_shape().as_list()
        Q = tf.contrib.layers.convolution(data, hidden_dim, 1)
        Q = tf.reshape(Q, [-1, t*h*w, hidden_dim])

        if pool:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.nn.max_pool3d(K, [1,2,2,1,1], strides=[1,2,2,1,1], padding='VALID')
            K = tf.reshape(K, [-1, (t//2)*(h//2)*w, hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.nn.max_pool3d(V, [1,2,2,1,1], strides=[1,2,2,1,1], padding='VALID')
            V = tf.reshape(V, [-1, (t//2)*(h//2)*w, hidden_dim])
        else:
            K = tf.contrib.layers.convolution(data, hidden_dim, 1)
            K = tf.reshape(K, [-1, t*h*w, hidden_dim])

            V = tf.contrib.layers.convolution(data, hidden_dim, 1)
            V = tf.reshape(V, [-1, t*h*w, hidden_dim])

        attention = scaled_dot_product(Q, K)
        output = tf.matmul(attention, V)  # [batch_size, sequence_length, output_dim]
        output = tf.reshape(output, [-1,t,h,w,hidden_dim])
        output = tf.contrib.layers.convolution(output, output_dim, 1)
        output = output + data  # Residual

        return output

def multiheaded_attention(data, hidden_dim, att_output_dim, output_dim, num_attention_layer=8):
    each_attention = []
    for _ in range(num_attention_layer):
        output = self_attention(data, hidden_dim, att_output_dim, residual=False)
        each_attention.append(output)

    output = tf.concat(each_attention, axis=1)
    output = tf.layers.dense(output, output_dim)
    return output
