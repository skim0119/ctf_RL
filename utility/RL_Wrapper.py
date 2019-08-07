""" NN Model Wrapper """

import numpy as np

import tensorflow as tf

from utility.utils import store_args
from utility.utils import discount_rewards

from method.base import initialize_uninitialized_vars as iuv


class Subgraph:
    """Subgraph

    Import pre-trained model for simulation and training.
    
    It does not initialize model.
    """
    @store_args
    def __init__(
        self,
        in_size,
        model_name,
        network_type=None,
        sess=None,
        global_network=None,
        record=False,
        *args,
        **kwargs
    ):
        if sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
        else:
            self.graph = self.sess.graph

        with self.sess.as_default(), self.graph.as_default():
            self.network = network_type(
                in_size=in_size,
                action_size=5,
                lr_actor=5e-5,
                lr_critic=2e-4,
                scope=model_name,
                entropy_beta = 0.01,
                sess=sess,
                global_network=global_network
            )
            self.network.initialize_vars()
            self.network.pull_global()


    def get_action(self, states):
        actions, values = self.network.run_network(states)
        return actions, values

    def train(self, trajs, bootstrap=0.0, gamma=0.98):
        buffer_s, buffer_a, buffer_tdtarget, buffer_adv = [], [], [], []
        for idx, traj in enumerate(trajs):
            if len(traj) == 0:
                continue
            observations = traj[0]
            actions = traj[1]
            rewards = np.array(traj[2])
            values = np.array(traj[3])
            
            value_ext = np.append(values, [bootstrap[idx]])
            td_target  = rewards + gamma * value_ext[1:]
            advantages = rewards + gamma * value_ext[1:] - value_ext[:-1]
            advantages = discount_rewards(advantages,gamma)
            
            buffer_s.extend(observations)
            buffer_a.extend(actions)
            buffer_tdtarget.extend(td_target.tolist())
            buffer_adv.extend(advantages.tolist())

        if len(buffer_s) == 0:
            return

        # Stack buffer
        feed_dict = {
            self.network.state_input : np.stack(buffer_s),
            self.network.action_ : np.array(buffer_a),
            self.network.td_target_ : np.array(buffer_tdtarget),
            self.network.advantage_ : np.array(buffer_adv),
        }

        # Update Buffer
        self.network.update_global(
            buffer_s,
            buffer_a,
            buffer_tdtarget,
            buffer_adv
        )

        # get global parameters to local ActorCritic 
        self.network.pull_global()
        
        return 
    


class TrainedNetwork:
    """TrainedNetwork

    Import pre-trained model for simulation only.
    
    It does not include any training sequence.
    """

    @store_args
    def __init__(
        self,
        model_name,
        input_tensor='global/state:0',
        output_tensor='global/actor/Softmax:0',
        action_space=5,
        sess=None,
        device=None,
        import_scope='',
        *args,
        **kwargs
    ):
        self.input_tensor = import_scope + '/' + input_tensor
        self.output_tensor = import_scope + '/' + output_tensor
        if sess is None:
            self.graph = tf.Graph()
            self.graph.device(device)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=self.graph)

        self.model_path = 'model/' + model_name

        # Initialize Session and TF graph
        self._initialize_network()

    def get_action(self, input_tensor):
        with self.sess.as_default(), self.sess.graph.as_default():
            feed_dict = {self.state: input_tensor}
            action_prob = self.sess.run(self.action, feed_dict)

        action_out = [np.random.choice(self.action_space, p=prob / sum(prob)) for prob in action_prob]

        return action_out

    def reset_network_weight(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise AssertionError

    def _initialize_network(self, verbose=False):
        """reset_network
        Initialize network and TF graph
        """
        def vprint(*args):
            if verbose:
                print(args)

        input_tensor = self.input_tensor
        output_tensor = self.output_tensor

        # Reset the weight to the newest saved weight.
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        vprint(f'path find: {ckpt.model_checkpoint_path}')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            vprint(f'path exist : {ckpt.model_checkpoint_path}')
            with self.sess.graph.as_default():
                self.saver = tf.train.import_meta_graph(
                    ckpt.model_checkpoint_path + '.meta',
                    clear_devices=True,
                    import_scope=self.import_scope
                )
                self.saver.restore(self.sess, ckpt.model_checkpoint_path, )
                vprint([n.name for n in self.sess.graph.as_graph_def().node])

                self.state = self.sess.graph.get_tensor_by_name(input_tensor)

                try:
                    self.action = self.sess.graph.get_operation_by_name(output_tensor)
                except ValueError:
                    self.action = self.sess.graph.get_tensor_by_name(output_tensor)
                    vprint([n.name for n in self.sess.graph.as_graph_def().node])

            vprint('Graph is succesfully loaded.', ckpt.model_checkpoint_path)
            #iuv(self.sess)
        else:
            vprint('Error : Graph is not loaded')
            raise NameError

    def _get_node(self, name):
        try:
            node = self.sess.graph.get_operation_by_name(name)
        except ValueError:
            node = self.sess.graph.get_tensor_by_name(name)
        return node
