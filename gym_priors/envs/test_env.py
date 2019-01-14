#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:12:58 2019

@author: molano
"""
import sys
import gym
import gym_priors
import numpy as np
import argparse
adhf = argparse.ArgumentDefaultsHelpFormatter


def arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=adhf)
    parser.add_argument('--exp_dur', help='exp. duration in num. of trials',
                        type=int, default=100)
    parser.add_argument('--trial_dur', help='num. of steps in each trial',
                        type=int, default=10)
    parser.add_argument('--rew', help='rewards for: stop fix, fix, hit, fail',
                        type=list, default=(-0.1, 0.0, 1.0, -1.0))
    parser.add_argument('--block_dur', help='num. of trials x block', type=int,
                        default=200)
    parser.add_argument('--stim_ev', help='level of difficulty of the exp.',
                        type=float, default=0.5)
    parser.add_argument('--rep_prob', help='rep. prob. for each block',
                        type=list, default=(.2, .8))
    parser.add_argument('--folder', help='where to save the data',
                        type=str, default='')
    return parser


def make_env(args):
    env = gym.make('priors-v0')
    arg_pars = arg_parser()
    params, unk_params = arg_pars.parse_known_args(args)
    print('what is this?:')
    print(unk_params)
    print(params)
    env.update_params(params)

#    env.exp_dur = 2

#    env.step(0)
#
#    for ind_stp in range(1000):
#        env.step(np.random.choice([0, 1, 2]))


if __name__ == '__main__':
    print(sys.argv)
    make_env(sys.argv)
