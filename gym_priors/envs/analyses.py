#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:51:01 2018

@author: molano
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplot
import os
import glob
plt.close('all')
colors = 'wy'


def plot_trials(folder, train, trial, tr_per=[0, 100], gng_time=0):
    aux = np.load(folder+'train_'+train+'/all_points_'+trial+'.npz')
    new_tr_flag = aux['new_trial_flags']
    plt.figure(figsize=(8, 8), dpi=250)

    # plot the stimulus
    subplot(7, 1, 1)
    states = aux['states']
    shape_aux = (states.shape[0], states.shape[2])
    states = np.reshape(states, shape_aux)[tr_per[0]:tr_per[1], :]
    plt.imshow(states.T, aspect='auto', cmap='gray')
    minimo = np.min(-0.5)
    maximo = np.max(shape_aux[1]-0.5)
    if gng_time == 0:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per)
    else:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per, color=colors[0])
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per,
                          const=gng_time, color=colors[1])
    plt.ylabel('stim')
    plt.xticks([])
    plt.yticks([])

    # subplot(6,1,2)
    # leave this subplot empty for a task-specific figure

    # plot actions
    actions = aux['actions'][tr_per[0]:tr_per[1]]
    actions = np.reshape(actions, (1, -1))
    minimo = np.min(-0.5)
    maximo = np.max(0.5)
    subplot(7, 1, 3)
    plt.imshow(actions, aspect='auto', cmap='viridis')
    if gng_time == 0:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per)
    else:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per, color=colors[0])
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per,
                          const=gng_time, color=colors[1])
    plt.ylabel('action')
    plt.xticks([])
    plt.yticks([])
    # plot the rewards
    rewards = aux['rewards'][tr_per[0]:tr_per[1]]
    rewards = np.reshape(rewards, (1, -1))
    minimo = np.min(-0.5)
    maximo = np.max(0.5)
    subplot(7, 1, 4)
    plt.imshow(rewards, aspect='auto', cmap='jet')
    if gng_time == 0:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per)
    else:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per, color=colors[0])
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per,
                          const=gng_time, color=colors[1])
    plt.ylabel('reward')
    plt.xticks([])
    plt.yticks([])
    # plot the performance
    performance = np.array(aux['corrects'])[tr_per[0]:tr_per[1], :]
    performance = performance.T
    minimo = np.min(-0.5)
    maximo = np.max(1.5)
    subplot(7, 1, 6)
    plt.imshow(performance, aspect='auto', cmap='jet')
    if gng_time == 0:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per)
    else:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per, color=colors[0])
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per,
                          const=gng_time, color=colors[1])
    plt.ylabel('correct')
    plt.xticks([])
    plt.yticks([])
    # plot neurons' activities
    activity = aux['net_state']
    subplot(7, 1, 7)
    shape_aux = (activity.shape[0], activity.shape[2])
    activity = np.reshape(activity, shape_aux)[tr_per[0]:tr_per[1], :]
    maximo = np.max(activity, axis=0).reshape(1, activity.shape[1])
    activity /= maximo
    activity[np.isnan(activity)] = -0.1
    plt.imshow(activity.T, aspect='auto', cmap='hot')
    minimo = np.min(-0.5)
    maximo = np.max(shape_aux[1]-0.5)
    if gng_time == 0:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per)
    else:
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per, color=colors[0])
        plot_trials_start(new_tr_flag, minimo, maximo, tr_per,
                          const=gng_time, color=colors[1])
    plt.ylabel('activity')
    plt.xlabel('time (a.u)')
    plt.yticks([])


def plot_trials_start(trials, minimo, maximo, tr_per, const=0, color='k'):
    trials = np.nonzero(trials)[0] + const - 0.5
    cond = np.logical_and(trials >= tr_per[0], trials <= tr_per[1])
    trials = trials[np.where(cond)]
    for ind_tr in range(len(trials)):
        plt.plot([trials[ind_tr], trials[ind_tr]], [minimo, maximo],
                 '--'+color, lw=0.5)
    plt.xlim(tr_per[0]-0.5, tr_per[1]-0.5)


def get_latest_working_file(folder, ind_worker):
    folder_aux = folder + 'trains/train_' + str(ind_worker) + '/trials_stats_*'
    all_files = glob.glob(folder_aux)
    numbers = get_list_of_numbers_in_files(all_files, 'trials_stats_')
    numbers = np.sort(numbers)[::-1]
    success = False
    data = []
    if len(numbers) == 0:
        print('there are no data files associated with the provided folder:')
        print(folder_aux)
    # go over all files and get the latest, working one.
    # the non-working are renamed
    for ind_f in range(len(numbers)):
        try:
            data = np.load(folder + 'trains/train_' + str(ind_worker) +
                           '/trials_stats_'+str(int(numbers[ind_f]))+'.npz')
            success = True
            break
        except IOError:
            os.rename(folder + 'trains/train_' + str(ind_worker) +
                      '/trials_stats_' + str(int(numbers[ind_f])) + '.npz',
                      folder + 'trains/train_' + str(ind_worker) +
                      '/NOTVALID_'+str(int(numbers[ind_f]))+'.npz')

    return data, success


def get_list_of_numbers_in_files(files, name):
    '''
    finds the number in a list of files so as to order them
    '''
    numbers = []
    for ind in range(len(files)):
        file = files[ind]
        aux = file[file.find(name)+len(name):file.find('npz')-1]
        numbers.append(float(aux))

    return numbers


if __name__ == '__main__':
    folder = '/home/molano/expectations_project/dual_task/' +\
        'trial_duration_20_rewards_-0.1_0.0_1.0_-1.0_noise_1.0' +\
        '_gamma_0.8_num_units_64_update_net_step_100/trains/'

    train = '1'
    trial = '20100'
    plot_trials(folder, train, trial)
