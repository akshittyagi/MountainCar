import os
import sys
import random
import argparse
import pickle as pkl
import csv
from enum import Enum
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

class MDP(object):

    def __init__(self, order):
        self.actions = {'forward': 1, 'neutral': 0, 'reverse': -1}
        self.gamma = 1
        self.phi = []
        self.states = 3
        space = order + 1
        for idx0 in range(space):
            for idx1 in range(space):
                for idx2 in range(space):
                    curr_c = np.array([idx0, idx1, idx2]).reshape(1, 3)
                    self.phi.append(curr_c)
        self.phi = np.array(self.phi)
     
    def get_q_value_function(self, k, weight, state, action):
        self.space = k+1
        curr_s = [state['x'], state['v']]
        curr_s.append(action + 1)
        curr_s = np.array(curr_s).reshape(len(curr_s), 1)
        phi = self.phi.dot(curr_s)
        phi = np.array(phi).reshape(len(phi),)
        return weight.T.dot(phi), phi

    
    def get_init_state(self):
        return {'x': -0.5, 'v': 0}
    
    def is_terminal_state(self, state):
        if state['x'] >= 0.5:
            return True
        return False
    
    def policy(self, policy, state):
        pass
    
    def transition_function(self, state, action):
        x_t, v_t = state['x'], state['v']
        v_t_1 = v_t + 0.001*action - 0.0025*np.cos(3*x_t)
        x_t_1 = x_t + v_t_1
        if x_t_1 <= -1.2 or x_t_1 >= 0.5:
            v_t_1 = 0
        return {'x': x_t_1, 'v': v_t_1}
    
    def reward_function(self, s_t, a_t, s_t_1):
        if s_t['x'] < 0.5 or s_t['x'] >= -1.2:
            return -1
        else:
            return 0
    
    def run_episode(self, policy):
        pass

