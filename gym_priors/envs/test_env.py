#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:12:58 2019

@author: molano
"""

import gym
import gym_priors
import numpy as np

env = gym.make('priors-v0')

env.exp_dur = 2

env.step(0)
#
#for ind_stp in range(102):
#  print(env.step(np.random.choice([0, 1, 2])))
