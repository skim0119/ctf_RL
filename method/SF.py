import tensorflow as tf
import numpy as np

from utility.utils import store_args

from network.model_SF import PPO_SF


class Network:
    @store_args
    def __init__(
        self,
        input_shape,
        action_size,
        scope,
        lr=1e-4,
        sess=None,
        N=16,
        **kwargs
    ):
        assert sess is not None, "TF Session is not given."

        with self.sess.as_default(), self.sess.graph.as_default():
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(shape=input_shape, dtype=tf.float32, name='state')
                self.action_ = tf.placeholder(shape=[None], dtype=tf.int32, name='action_hold')
                self.td_target_ = tf.placeholder(shape=[None, N], dtype=tf.float32, name='td_target_hold')
                self.reward_ = tf.placeholder(shape=[None], dtype=tf.int32, name='reward_hold')
                self.advantage_ = tf.placeholder(shape=[None], dtype=tf.float32, name='adv_hold')
                self.old_logits_ = tf.placeholder(shape=[None, action_size], dtype=tf.float32, name='old_logit_hold')
                self.done_state_ = tf.placeholder(shape=[None], dtype=tf.float32, name='done_hold')

                # Build Network
                model = PPO_SF(action_size, phi_n=N);  self.model = model
                inputs = self.state_input
                self.actor, self.logits, self.log_logits, self.critic, self.phi, self.sf_reward, self.psi = model(inputs)
                actor_loss, sf_loss, reward_loss = model.build_loss(self.old_logits_, self.action_, self.advantage_, self.td_target_, self.reward_)
                model.summary()

                # Build Trainer
                actor_optimizer = tf.keras.optimizers.Adam(lr)
                actor_grad = actor_optimizer.get_gradients(actor_loss, model.get_actor_variables)
                actor_update = actor_optimizer.apply_gradients(zip(actor_grad, model.get_actor_variables))

                sf_optimizer = tf.keras.optimizers.Adam(lr*1e-1)
                sf_grad = sf_optimizer.get_gradients(sf_loss, model.get_psi_variables)
                sf_update = sf_optimizer.apply_gradients(zip(sf_grad, model.get_psi_variables))

                reward_optimizer = tf.keras.optimizers.Adam(lr*1e-1)
                reward_grad = reward_optimizer.get_gradients(reward_loss, model.get_phi_variables)
                reward_update = reward_optimizer.apply_gradients(zip(reward_grad, model.get_phi_variables))

                self.gradients = actor_grad + sf_grad + reward_grad
                self.update_rl = tf.group([actor_update, sf_update])
                self.update_sl = reward_update

    def run_network(self, states, return_action=True):
        feed_dict = {self.state_input: states}
        query = [self.actor, self.critic, self.log_logits, self.phi, self.psi]
        a_probs, critics, logits, phi, psi = self.sess.run(query, feed_dict)
        if return_action:
            actions = np.array([np.random.choice(self.action_size, p=prob / sum(prob)) for prob in a_probs])
            return actions, critics, logits, phi, psi
        else:
            actions = np.array([0 for prob in a_probs])
            return actions, critics, logits, phi, psi
            #return a_probs, critics, logits, phi, psi

    def update_network(self, state_input, action, td_target, advantage, old_logit, state_next, reward, global_episodes, writer=None, log=False):
        feed_dict = {self.state_input: state_input,
                     self.action_: action,
                     self.td_target_: td_target,
                     self.reward_: reward,
                     self.advantage_: advantage,
                     self.old_logits_: old_logit}
        self.sess.run(self.update_rl, feed_dict)


        feed_dict[self.state_input] = state_next
        self.sess.run(self.update_sl, feed_dict)

        if log:
            feed_dict[self.state_input] = state_input
            ops = [self.model.actor_loss, self.model.critic_loss, self.model.reward_loss, self.model.entropy]
            aloss, closs, rloss, entropy = self.sess.run(ops, feed_dict)

            summary = tf.Summary()
            summary.value.add(tag='summary/actor_loss', simple_value=aloss)
            summary.value.add(tag='summary/critic_loss', simple_value=closs)
            #summary.value.add(tag='summary/reward_mse_loss', simple_value=rloss)
            summary.value.add(tag='summary/entropy', simple_value=entropy)

            # Check vanish gradient
            grads = self.sess.run(self.gradients, feed_dict)
            total_counter = 0
            vanish_counter = 0
            for grad in grads:
                total_counter += np.prod(grad.shape) 
                vanish_counter += (np.absolute(grad)<1e-8).sum()
            summary.value.add(tag='summary/grad_vanish_rate', simple_value=vanish_counter/total_counter)
            
            writer.add_summary(summary,global_episodes)

            writer.flush()

    @property
    def get_successor_feature_gradcam(self):
        with tf.name_scope('gradcam_module'):
            scope = self.state_input 
            phi_gradcam = [tf.gradients(self.phi[:,i], self.state_input)[0] for i in range(self.N)]
            psi_gradcam = [tf.gradients(self.psi[:,i], self.state_input)[0] for i in range(self.N)]
        return phi_gradcam, psi_gradcam

    @property
    def get_successor_weight(self):
        return self.model.successor_layer.weights

    def initialize_vars(self):
        var_list = self.get_vars
        init = tf.initializers.variables(var_list)
        self.sess.run(init)

    def initiate(self, saver, model_path):
        # Restore if savepoint exist. Initialize everything else
        with self.sess.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Load Model : ", ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())
                print("Initialized Variables")

    def save(self, saver, model_path, global_step):
        saver.save(self.sess, model_path, global_step=global_step)

    def load(self, model_path):
        # Restore if savepoint exist. Error if checkpoint does not exist
        with self.sess.graph.as_default():
            ckpt = tf.train.latest_checkpoint(model_path)
            self.model.load_weights(ckpt)
        print("Load Model : ", ckpt)

    @property
    def get_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        

