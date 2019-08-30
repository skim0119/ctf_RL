"""Simple agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""

import numpy as np
from collections import defaultdict
from utility.RL_Wrapper import TrainedNetwork
from utility.dataModule import oh_to_rgb

import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from scipy import ndimage

from policy.policy import Policy

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
mycmap = transparent_cmap(plt.cm.gray)

class PPO(Policy):
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
        policy: Method to determine the action based on observation for a single unit
        scan : Method that returns the dictionary of object around the agent

    Variables:
        exploration : exploration rate
        previous_move : variable to save previous action
    """

    def unstack_frame(frames):
        s = np.concatenate(frames, axis=3)
        return s

    def append_frame(l:list, obj):
        l.append(obj)
        l.pop(0)
        assert len(l) == keep_frame

    def __init__(self):
        self.keep_frame = 4
        self.vision_range = 19
        self.network = TrainedNetwork(
                #model_name='../raide_rl/model/ppo_flat_robust',
                model_name='model/imitate_baseline',
                #model_name='model/ppo_var1',
                input_tensor='main/state:0',
                #output_tensor='main/actor/Softmax:0'
                output_tensor='main/PPO/activation/Softmax:0'
            )

    def initiate(self, free_map, agent_list):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.free_map = free_map
        self.agent_list = agent_list

        self.initial_move = True
        self.stacked_frame = None


    def gen_action(self, agent_list, observation):
        """Action generation method.

        This is a required method that generates list of actions corresponding
        to the list of units.

        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).

        Returns:
            action_out (list): list of integers as actions selected for team.

        """

        inputx, inputy = 39, 39
        obs = self.center_pad(observation, width=self.vision_range)

        if self.initial_move:
            self.stacked_frame = [obs for _ in range(self.keep_frame)]
            self.initial_move = False
        else:
            self.stacked_frame.pop(0)
            self.stacked_frame.append(obs)
        obs = np.concatenate(self.stacked_frame, axis=2)

        agent_state = []
        for idx, agent in enumerate(agent_list):
            x, y = agent.get_loc()
            agent_state.append(obs[x:x+inputx, y:y+inputy, :])
        state = np.stack(agent_state)
        action_out = self.network.get_action(state)

        return action_out

    def center_pad(self, m, width, padder=[0,0,0,1,0,0]):
        lx, ly, nch = m.shape
        pm = np.zeros((lx+(2*width),ly+(2*width), nch), dtype=np.int)
        for ch, pad in enumerate(padder):
            pm[:,:,ch] = pad
        pm[width:lx+width, width:ly+width] = m
        return pm

class PPO_visualize(Policy):
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
        policy: Method to determine the action based on observation for a single unit
        scan : Method that returns the dictionary of object around the agent

    Variables:
        exploration : exploration rate
        previous_move : variable to save previous action
    """

    def unstack_frame(frames):
        s = np.concatenate(frames, axis=3)
        return s

    def append_frame(l:list, obj):
        l.append(obj)
        l.pop(0)
        assert len(l) == keep_frame

    def __init__(self):
        self.keep_frame = 4
        self.vision_range = 19
        self.network = TrainedNetwork(
                #model_name='../raide_rl/model/ppo_flat_robust',
                model_name='model/adapt_train/ppo_flat',
                input_tensor='main/state:0',
                output_tensor='main/actor/Softmax:0'
            )

        style.use('fivethirtyeight')
        self.fig = plt.figure(figsize=(16,8))
        self.ax = [
                self.fig.add_subplot(2,4,3),
                self.fig.add_subplot(2,4,4),
                self.fig.add_subplot(2,4,7),
                self.fig.add_subplot(2,4,8),
                ]
        self.big_ax = self.fig.add_subplot(2,4,(1,6))
        #self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax[0].set_axis_off()
        self.ax[1].set_axis_off()
        self.ax[2].set_axis_off()
        self.ax[3].set_axis_off()
        self.big_ax.set_axis_off()

        self.graph_data = []
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=100)
        plt.show(block=False)

    def animate(self, i):
        if len(self.graph_data) == 2:
            feature_map, gradcams = self.graph_data
        else: 
            return
        def conv_heat_map(output, grads_val):
            weights = np.mean(grads_val, axis = (0, 1)) # alpha_k
            
            cam = np.zeros(output.shape[:2], dtype = np.float32)
        
            # Taking a weighted average
            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

            # ReLU
            #cam = np.abs(cam-np.mean(cam))
            m = min(cam[0,0], cam[-1,0])
            cam = np.maximum(cam-m, 0)
            cam = cam / (np.max(cam)-np.min(cam)) # scale 0 to 1.0
            cam = resize(cam, (351,351), preserve_range=True)
            #cam = np.repeat(cam, 9, axis=0)
            #cam = np.repeat(cam, 9, axis=1)

            cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
            return cam_heatmap
        self.big_ax.clear()
        
        if len(self.graph_data) == 2:
            #state = resize(oh_to_rgb(feature_map[:1,:,:,-6:])[0]/255, (360, 360), preserve_range=True)
            state = oh_to_rgb(feature_map[:1,:,:,-6:])[0]/255
            state = np.repeat(state, 9, axis=0)
            state = np.repeat(state, 9, axis=1)
            for ax, gradcam, title in zip(self.ax, gradcams, self.graph_title):
                ax.clear()
                #self.ax1.plot(xs,ys)
                heatmap = conv_heat_map(feature_map[0], gradcam[0])
                ax.imshow(state, alpha=0.5)
                ax.imshow(heatmap, alpha=0.75, vmin=min(heatmap.flat), vmax=max(heatmap.flat))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=8)
            self.big_ax.imshow(state)
            self.big_ax.set_xticks([])
            self.big_ax.set_yticks([])
            self.big_ax.set_title('env', fontsize=8)

    def initiate(self, free_map, agent_list):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.free_map = free_map
        self.agent_list = agent_list

        self.initial_move = True
        self.stacked_frame = None

        self.graph_data = []

    def gen_action(self, agent_list, observation):
        """Action generation method.

        This is a required method that generates list of actions corresponding
        to the list of units.

        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).

        Returns:
            action_out (list): list of integers as actions selected for team.

        """

        inputx, inputy = 39, 39
        obs = self.center_pad(observation, width=self.vision_range)

        if self.initial_move:
            self.stacked_frame = [obs for _ in range(self.keep_frame)]
            self.initial_move = False
        else:
            self.stacked_frame.pop(0)
            self.stacked_frame.append(obs)
        obs = np.concatenate(self.stacked_frame, axis=2)

        agent_state = []
        for idx, agent in enumerate(agent_list):
            x, y = agent.get_loc()
            agent_state.append(obs[x:x+inputx, y:y+inputy, :])
        state = np.stack(agent_state)
        action_out, critic, feature_map, gradcams = self.network.get_action(state)
        self.graph_title = []
        for n, w in zip(self.network.cr_index, self.network.critic_w[self.network.cr_index]):
            self.graph_title.append('node {}: w={:.3f}'.format(n,w))

        self.graph_data = (feature_map, gradcams)

        return action_out

    def center_pad(self, m, width, padder=[1,0,0,1,0,0]):
        lx, ly, nch = m.shape
        pm = np.zeros((lx+(2*width),ly+(2*width), nch), dtype=np.int)
        for ch, pad in enumerate(padder):
            pm[:,:,ch] = pad
        pm[width:lx+width, width:ly+width] = m
        return pm

