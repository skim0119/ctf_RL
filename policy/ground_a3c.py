""" Reinforce Learning Policy Generator

This module calls any pre-trained tensorflow model and generates the policy.
It includes method to switch between the network and weights.

For use: generate policy to simulate along with CTF environment.

for Capture the Flag environment.
    http://github.com/osipychev/missionplanner/

DOs/Denis Osipychev
    http://www.denisos.com

Last Modified:
    Seung Hyun Kim
    created :Wed Oct 24 12:21:34 CDT 2018
"""

import numpy as np
import tensorflow as tf
from utility.dataModule import one_hot_encoder as one_hot_encoder
from utility.utils import store_args


class PolicyGen:
    """Policy generator class for CtF env.

    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action  : Required method to generate a list of actions.
        load_model  : Load pre-defined model (*.meta file). Only TensorFlow model supported
        load_weight : Load/reload weight to the model.
    """

    @store_args
    def __init__(self,
                 free_map=None,
                 agent_list=None,
                 model_dir='./model/partial/',
                 input_name='global/state:0',
                 output_name='global/actor/fully_connected_1/Softmax:0',
                 import_scope=None,
                 *args,
                 **kwargs
                 ):
        """Constuctor for policy class.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.

        Initialize TensorFlow Graph
        Initiate session
        """

        self.input_shape = 19
        self.reset_network()

    def gen_action(self, agent_list, observation, free_map=None):
        """Action generation method.

        This is a required method that generates list of actions corresponding
        to the list of units.

        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).

        Returns:
            action_out (list): list of integers as actions selected for team.

        Note:
            The graph is not updated in this session.
            It only returns action for given input.
        """

        obs = one_hot_encoder(observation, agent_list, self.input_shape)
        action_prob = self.sess.run(self.action, feed_dict={self.state: obs})  # Action Probability

        action_out = [np.random.choice(5, p=action_prob[x] / sum(action_prob[x])) for x in range(len(agent_list))]

        return action_out

    def reset_network(self, input_name=None, output_name=None, scope=None):
        """reset_network
        Initialize network and TF graph
        """
        if input_name is None:
            input_name = self.input_name

        if output_name is None:
            output_name = self.output_name

        # Reset the weight to the newest saved weight.
        print(f'policy initialization : path {self.model_dir}')
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        print(f'path find: {ckpt.model_checkpoint_path}')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print(f'path exist : {ckpt.model_checkpoint_path}')
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf.Session()
                self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', import_scope=scope, clear_devices=True)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                #print([n.name+'\n' for n in self.graph.as_graph_def().node if n.name[0]=='g'])

                self.state = self.graph.get_tensor_by_name(input_name)
                self.encoded_state = self.graph.get_tensor_by_name('global/actor/Flatten/flatten/Reshape:0')
                try:
                    self.action = self.graph.get_operation_by_name(output_name)
                except ValueError:
                    self.action = self.graph.get_tensor_by_name(output_name)
                    #print([n.name for n in self.graph.as_graph_def().node])

            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            print('Error : Graph is not loaded')
            raise NameError

    def query_encoded_state(self, state):
        feed_dict = {self.state: state}
        query_ops = self.encoded_state
        return self.sess.run(query_ops, feed_dict=feed_dict)
