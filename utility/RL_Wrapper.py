""" NN Model Wrapper """

import numpy as np

import tensorflow as tf

from utility.utils import store_args

from utility.utils import discount_rewards


class Model:
    """
    Model wraps the existing method and provide common methods.
        - train
        - save
        - load
        - run
    """

    @store_args
    def __init__(
        self,
        model_path=None,
        log_path=None,
        name=None,
        sess=None,
        prohibit_log=False,
        *args,
        **kwargs
    ):
        """ __init__
        Parameters
        ----------------

        prohibit_log : [bool] 
            If true, any log in this model will be ignored.
            It mainly prohibit logging in train
        """
        # Assertions
        assert model_path is not None, 'model path is not specified'
        assert log_path is not None, 'model path is not specified'
        assert sess is not None and name is None, 'If pre-defiend session is used, must provide name of this network'
        assert name == 'global', 'Model name cannot be global'
       
        # Presets
        if name is not None:
            input_tensor = name + '/'  + input_tensor
            output_tensor = name + '/'  + output_tensor

        # Find checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        if not ckpt or not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise NameError('Error : Graph failed to load')
        else:
            print('Checkpoint Found.')
            print('Call TF pretrained model:')
            print('    checkpoint_path : {}'.format(ckpt.model_checkpoint_path))
            print('    input_name : {}'.format(input_name))
            print('    output_name : {}'.format(output_name))

        # Set TF graph and session
        if sess is None:
            config = tf.ConfigProto(device_count = {'GPU': 0})  # Only use CPU
            self.graph = tf.Graph()
            self.sess = tf.Session(config=config, graph=self.graph)
        else:  # Use given graph
            self.graph = sess.graph

        # Load and restore TF model
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(
                    meta_graph_or_file=ckpt.model_checkpoint_path+'.meta',
                    clear_devices=True,
                    import_scope=name,
                )
            iuv(self.sess)  # Initialize all undefined weights
        self.state, self.action = self.reset_network_weight()
        print('    TF policy loaded. {}'.format(name) )

    def run_network(self, input_tensor):
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state: input_tensor}
            action_prob = self.sess.run(self.action, feed_dict)

        return action_prob

    def reset_network_weight(self):
        """
        Reload the weight from the TF meta data
        """
        input_name = self.input_name
        output_name = self.output_name
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            state = self.graph.get_tensor_by_name(input_name)
            try:
                action = self.graph.get_operation_by_name(output_name)
            except ValueError:
                action = self.graph.get_tensor_by_name(output_name)
        return state, action


class TrainedNetwork:
    """TrainedNetwork

    Import pre-trained model for simulation only.
    
    - It does not include any training sequence.
    - It provides method to reload network weights.
    - It does not alter the network.
    """

    @store_args
    def __init__(
        self,
        model_path=None,
        input_tensor='global/state:0',
        output_tensor='global/actor/Softmax:0',
        name=None,
        sess=None,
        *args,
        **kwargs
    ):
        # Assertions
        assert model_path is not None, 'model path is not specified'
        if sess is not None: assert name is not None, 'If pre-defiend session is used, must provide name of this network'
        assert name != 'global', 'Model name cannot be global'
       
        # Presets
        if name is not None:
            input_tensor = name + '/'  + input_tensor
            output_tensor = name + '/'  + output_tensor

        # Find checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        if not ckpt or not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise NameError('Error : Graph failed to load')
        else:
            print('Checkpoint Found.')
            print('Call TF pretrained model:')
            print('    checkpoint_path : {}'.format(ckpt.model_checkpoint_path))
            print('    input_name : {}'.format(input_tensor))
            print('    output_name : {}'.format(output_tensor))

        # Set TF graph and session
        if sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
        else:  # Use given graph
            self.graph = sess.graph

        # Load and restore TF model
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(
                    meta_graph_or_file=ckpt.model_checkpoint_path+'.meta',
                    clear_devices=True,
                    import_scope=name,
                )
        self.state, self.action = self.reset_network_weight()
        print('    TF policy loaded. {}'.format(name) )

    def run_network(self, input_tensor):
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state: input_tensor}
            action_prob = self.sess.run(self.action, feed_dict)

        return action_prob

    def reset_network_weight(self):
        """
        Reload the weight from the TF meta data
        """
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            state = self.graph.get_tensor_by_name(self.input_tensor)
            try:
                action = self.graph.get_operation_by_name(self.output_tensor)
            except ValueError:
                action = self.graph.get_tensor_by_name(self.output_tensor)
        return state, action

