import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import IPython
e = IPython.embed
import random
from collections import namedtuple
import h5py

State = namedtuple("State", "observation reward")

class Mock_LG_ENV:
    def __init__(self):
        self.actions, self.images, self.qposes = None, None, None
        self.idx = 0

    def reset(self, ep=None):
        if ep is None:
            ep = random.randint(0, 31)
        with h5py.File(f'/home/jellyho/LG/act/LG_Dataset/episode_{ep}.hdf5', 'r') as root:
            self.actions = root['/action'][()]
            self.images = root['/observations/images/camera'][()]
            self.qposes = root['/observations/qpos'][()]
        self.idx = 0
        self.max_timesteps = len(self.images)
        obs = {
            'images': {
                'camera' : self.images[self.idx]
            },
            'qpos':self.qposes[self.idx]
        }
        output = State(obs, 0)
        return output
    
    def step(self, action):
        reward = (- (self.actions[self.idx] - action) ** 2).sum()
        self.idx += 1
        done = self.idx >= self.max_timesteps - 1

        obs = {
            'images': {
                'camera' : self.images[self.idx]
            },
            'qpos':self.qposes[self.idx]
        }
        output = State(obs, reward)
        return output

