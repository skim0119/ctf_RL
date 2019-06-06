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
                 model_dir='./policy/A3C_model/',
                 input_name='global/state:0',
                 output_name='global/actor/fully_connected_1/Softmax:0',
                 import_scope=None,
                 vision_radius=19,
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

        # Switches
        self.full_observation = True

        self._reset_done = False

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
        if not self._reset_done:
            self.reset_network_weight()

        obs = one_hot_encoder(state=observation,
                agents=agent_list, vision_radius=self.vision_radius)
        logit = self.sess.run(self.action, feed_dict={self.state: obs})  # Action Probability
        action_out = [np.random.choice(5, p=logit[x] / sum(logit[x])) for x in range(len(agent_list))]

        return action_out

    def reset_network_weight(self, input_name=None, output_name=None):
        if not self._reset_done:
            self.reset_network()
            self._reset_done = True
        else:
            if input_name is None:
                input_name = self.input_name
            if output_name is None:
                output_name = self.output_name
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            with self.sess.graph.as_default():
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', import_scope=None, clear_devices=True)
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise AssertionError

    def reset_network(self, input_name=None, output_name=None, scope=None):
        """reset_network
        Initialize network and TF graph
        """
        if input_name is None:
            input_name = self.input_name

        if output_name is None:
            output_name = self.output_name

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        # Reset the weight to the newest saved weight.
        print(f'policy initialization : path {self.model_dir}')
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        print(f'path find: {ckpt.model_checkpoint_path}')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print(f'path exist : {ckpt.model_checkpoint_path}')
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf.Session(config=config)
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', import_scope=scope, clear_devices=True)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                #print([n.name for n in self.graph.as_graph_def().node])

                self.state = self.graph.get_tensor_by_name(input_name)
                try:
                    self.action = self.graph.get_operation_by_name(output_name)
                except ValueError:
                    self.action = self.graph.get_tensor_by_name(output_name)
                    #print([n.name for n in self.graph.as_graph_def().node])

            print('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
        else:
            print('Error : Graph is not loaded')
            raise NameError

