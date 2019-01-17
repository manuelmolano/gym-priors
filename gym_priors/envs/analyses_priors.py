#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:03:04 2018

@author: molano
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import analyses
import utils
# This is the 3D plotting toolkit
# from mpl_toolkits.mplot3d import Axes3D

# parameters for figures
left = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots
line_width = 2


def load_prior_data(folder, num_workers=12, save_data=True):
    """
    loads the trial-by-trial data from the experiment indicated by folder,
    put the data from all workers together and save it
    """
    stim2 = []  # mean of stim 2 (mean of stimulus 1 is always 1)
    trial_duration = []
    stims_pos = []  # postion of stim 1 (the rewarded stimulus)
    repeat = []  # whether stim 1 is repeating side
    reward_trials = []  # reward given at the end of the trial
    performance = []  # hit or fail at the end of the trial
    evidence = []  # total signed evidence(right - left evidence)
    repeating_prob = []
    net_state = []
    action = []
    min_num_trials = np.inf
    for ind_worker in range(num_workers):
        # go over all all workers and try to load latest data file
        data, success = analyses.get_latest_working_file(folder, ind_worker)
        if success:
            stim2.append(data['stim2'])
            trial_duration.append(data['trial_duration'])
            stims_pos.append(data['stims_position'])
            repeat.append(data['repeat'])
            reward_trials.append(data['reward'])
            performance.append(data['performance'])
            evidence.append(data['evidence'])
            repeating_prob.append(data['rep_prob'])
            if 'net_smmd_act' in data.keys():
                aux = data['net_smmd_act'].shape
                net_state.append(data['net_smmd_act'].reshape(aux[0], aux[2]))
            if 'action' in data.keys():
                action.append(data['action'])
            min_num_trials = np.min([len(data['stim2']), min_num_trials])

    # set all elements in the list to the same length
    stim2 = [x[:int(min_num_trials)] for x in stim2]
    trial_duration = [x[:int(min_num_trials)] for x in trial_duration]
    performance = [x[:int(min_num_trials)] for x in performance]
    stims_pos = [x[:int(min_num_trials)] for x in stims_pos]
    reward_trials = [x[:int(min_num_trials)] for x in reward_trials]
    repeat = [x[:int(min_num_trials)] for x in repeat]
    evidence = [x[:int(min_num_trials)] for x in evidence]
    repeating_prob = [x[:int(min_num_trials)] for x in repeating_prob]
    net_state = [x[:int(min_num_trials)] for x in net_state]
    action = [x[:int(min_num_trials)] for x in action]
    # transform to np.array
    stim2 = np.array(stim2)
    trial_duration = np.array(trial_duration)
    performance = np.array(performance)
    stims_pos = np.array(stims_pos)
    reward_trials = np.array(reward_trials)
    repeat = np.array(repeat)
    evidence = np.array(evidence)
    action = np.array(action)
    if len(net_state) > 0:
        net_state = np.array(net_state)
        net_state = np.transpose(net_state, (1, 0, 2))
        aux = net_state.shape
        net_state = net_state.reshape((aux[0], aux[1]*aux[2]))

    repeating_prob = np.array(repeating_prob)
    exp_setup = np.load(folder+'experiment_setup.npz')
    repeating_prob = exp_setup['repeating_prob'][repeating_prob]
    # save data
    if save_data:
        data = {'stim2': stim2, 'trial_duration': trial_duration,
                'performance': performance, 'stims_position': stims_pos,
                'reward_trials': reward_trials, 'repeat': repeat,
                'evidence': evidence, 'rep_prob': repeating_prob,
                'net_state': net_state, 'action': action}

        np.savez(folder + '/figures/trials_stats.npz', **data)

    return stim2, trial_duration, performance, stims_pos, reward_trials,\
        repeat, evidence, repeating_prob, net_state, action


def plot_cumulative_evidence(folder, train, trial, tr_per=[0, 100]):
    """
    plots the cumulative evidence for a given set of trials
    folder, train and trial identify the exp, worker, period, respectively
    tr_per: period of trials to be plot (allows reducing the num. of trials)
    """
    # load
    aux = np.load(folder+'train_'+train+'/all_points_'+trial+'.npz')
    # states corresponds to the stimuli delivered
    states = np.reshape(aux['states'], (aux['states'].shape[0], 3))
    # new trial flags
    trials_mat = aux['new_trial_flags']
    trials = np.nonzero(trials_mat)[0]
    trials = np.concatenate((np.array([-1]), trials))
    # go over trials and compute cumulative evidence
    evidence = np.zeros((int(np.diff(tr_per)),))
    for ind_time in range(tr_per[0], tr_per[1]):
        if ind_time in trials:
            # the cumulative evidence is 0 in the beginning
            evidence[ind_time] = 0
        else:
            # 0 belongs to trials so this is always valid
            previous_evidence = evidence[ind_time-1]
            evidence[ind_time] = previous_evidence +\
                (states[ind_time, 0]-states[ind_time, 1])

    # plot evidence
    plt.plot(evidence)
    plt.plot(tr_per, [0, 0], '--k', lw=0.5)
    plt.xlim(-0.5, 99.5)
    analyses.plot_trials_start(aux['new_trial_flags'],
                               np.min(evidence), np.max(evidence), tr_per)
    plt.ylabel('evidence')
    plt.xticks([])


def plot_learning(performance, evidence, stim_position, action,
                  folder='', name='', save_fig=False,
                  view_fig=False, f=False):
    """
    plots performance of the RNN and the ideal observer.
    it also plots the block (rep or alt) period and the choice of the RNN
    at each trial
    """

    # ideal observer choice
    io_choice = ideal_observer(evidence, stim_position)
    io_performance = io_choice == stim_position
    # save the mean performances
    RNN_perf = np.mean(performance[:, 2000:].flatten())
    io_perf = np.mean(io_performance[:, 2000:].flatten())
    if folder != '':
        data = {'RNN_perf': RNN_perf, 'io_perf': io_perf}
        np.savez(folder + '/performance' + name + '.npz', **data)

    if view_fig:
        if not f:
            f = plt.figure(figsize=(4, 8), dpi=250)
            matplotlib.rcParams.update({'font.size': 8})
            plt.subplots_adjust(left=left, bottom=bottom, right=right,
                                top=top, wspace=wspace, hspace=hspace)
        # network's choice
        # right_choice = action == 0
        w_conv = 200  # this is for the smoothing
        # plot right choice
        # label_aux = 'Right choice (' +\
        # str(round(np.mean(right_choice[:, 2000:].flatten()), 3)) + ')'
        # plt.plot(right_choice[0, :], color=(0.2, 1, 0.2), lw=0.5,
        # label=label_aux)
        # plot blocks
        # plt.plot(rep_probability, color=(0, 0, 1), lw=0.5, label='blocks')
        # plot smoothed performance
        performance_smoothed = np.convolve(np.mean(performance, axis=0),
                                           np.ones((w_conv,))/w_conv,
                                           mode='valid')
        plt.plot(performance_smoothed, color=(0.39, 0.39, 0.39), lw=0.5,
                 label='RNN perf. (' + str(round(RNN_perf, 3)) + ')')

        # plot ideal observer performance
        io_perf_smoothed = np.convolve(np.mean(io_performance, axis=0),
                                       np.ones((w_conv,))/w_conv,
                                       mode='valid')
        plt.plot(io_perf_smoothed, color=(1, 0.8, 0.5), lw=0.5,
                 label='Ideal Obs. perf. (' + str(round(io_perf, 3)) + ')')
        # plot 0.25, 0.5 and 0.75 performance lines
        # plot_fractions([0, performance.shape[1]])
        plt.title('performance')
        plt.xlabel('trials')

        print(folder)
        print(save_fig)
        print(f)
        if folder != '' and save_fig:
            f.savefig(folder + '/performance_' + name + '.svg',
                      dpi=600, bbox_inches='tight')
            plt.close(f)


def plot_psychometric_curves(evidence, performance, action,
                             blk_dur=200, stps=10**10, per=0,
                             folder='', name='', plt_av=True, figs=False):
    """
    plots psychometric curves
    - evidence for right VS prob. of choosing right
    - evidence for repeating side VS prob. of repeating
    - same as above but conditionated on hits and fails
    """

    # get periods within block to filter the data
    _, aux = np.mgrid[0:evidence.shape[0], 0:evidence.shape[1]]
    perds = np.floor((aux % blk_dur) / stps)

    # build the mat that indicates the current block
    rep_prob = build_block_mat(evidence.shape, blk_dur)

    # repeating probs. values
    probs_vals = np.unique(rep_prob)
    assert len(probs_vals) <= 2
    colors = [[1, 0, 0], [0, 0, 1]]
    if figs:
        rows = 2
        cols = 2
        f = plt.figure(figsize=(8, 8), dpi=250)
        matplotlib.rcParams.update({'font.size': 6})
        plt.subplots_adjust(left=left, bottom=bottom, right=right,
                            top=top, wspace=wspace, hspace=hspace)
    else:
        rows = 0
        cols = 0

    data = {}
    for ind_blk in range(len(probs_vals)):
        # filter data
        inds = (rep_prob == probs_vals[ind_blk]) *\
                (perds == per)
        evidence_block = evidence[inds]
        performance_block = performance[inds]
        action_block = action[inds]
        data = get_psyCho_curves_data(performance_block,
                                      evidence_block, action_block,
                                      probs_vals[ind_blk],
                                      rows, cols, figs, colors[ind_blk],
                                      plt_av, data)

    if folder != '':
        np.savez(folder + '/psychCurv_stats' +
                 name + '.npz', **data)
        if figs:
            f.savefig(folder+'/psychCurv_' + name + '.svg',
                      dpi=600, bbox_inches='tight')


def get_psyCho_curves_data(performance, evidence, action, prob,
                           rows, cols, figs, color, plt_av, data):

    # right-evidence_block VS prob. of choosing right
    # get the action
    right_choice = action == 0

    # associate invalid trials (network fixates) with random choice
    right_choice[action == 2] = evidence[action == 2] > 0
    # np.random.choice([0, 1], size=(np.sum(action.flatten() == 2),))

    # convert the choice to float and flatten it
    right_choice = [float(x) for x in right_choice]
    right_choice = np.asarray(right_choice)
    # fit and plot
    if figs:
        plt.subplot(rows, cols, 1)
        plt.xlabel('right evidence')
        plt.ylabel('prob. right')
    popt, pcov, av_data =\
        fit_and_plot(evidence, right_choice,
                     plt_av, color=color, figs=figs)

    data['popt_rightProb_' + str(prob)] = popt
    data['pcov_rightProb_' + str(prob)] = pcov
    data['av_rightProb_' + str(prob)] = av_data

    # repeating evidence VS prob. repeating
    # I add a random choice to the beginning of the choice matrix
    # and differentiate to see when the network is repeating sides
    repeat = np.concatenate(
            (np.array(np.random.choice([0, 1])).reshape(1,),
             right_choice))
    repeat = np.diff(repeat) == 0
    # right_choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    right_choice_repeating = np.concatenate(
            (np.array(np.random.choice([0, 1])).reshape(1, ),
             right_choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev_block = evidence *\
        (-1)**(right_choice_repeating == 0)
    # fitting
    if figs:
        label_aux = 'p. rep.: ' + str(prob)
        plt.subplot(rows, cols, 2)
        plt.xlabel('repetition evidence')
        plt.ylabel('prob. repetition')
    else:
        label_aux = ''
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block, repeat,
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_'+str(prob)] = popt
    data['pcov_repProb_'+str(prob)] = pcov
    data['av_repProb_'+str(prob)] = av_data

    # plot psycho-curves conditionated on previous performance
    # get previous trial performance
    prev_perf = np.concatenate(
            (np.array(np.random.choice([0, 1])).reshape(1,),
             performance[:-1]))
    # plot psycho-curves conditionated on previous correct
    # fitting
    mask = prev_perf == 1
    if figs:
        plt.subplot(rows, cols, 3)
        plt.xlabel('repetition evidence')
        plt.ylabel('prob. repetition')
        plt.title('Prev. hit')
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block[mask], repeat[mask],
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_hits_'+str(prob)] = popt
    data['pcov_repProb_hits_'+str(prob)] = pcov
    data['av_repProb_hits_'+str(prob)] = av_data

    # plot psycho-curves conditionated on previous wrong
    # fitting
    mask = prev_perf == 0
    if figs:
        plt.subplot(rows, cols, 4)
        plt.xlabel('repetition evidence')
        plt.ylabel('prob. repetition')
        plt.title('Prev. fail')
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block[mask], repeat[mask],
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_fails_'+str(prob)] = popt
    data['pcov_repProb_fails_'+str(prob)] = pcov
    data['av_repProb_fails_'+str(prob)] = av_data

    return data


def fit_and_plot(evidence, choice, plt_av=False,
                 color=(0, 0, 0), label='', figs=False):
    """
    uses curve_fit to fit the evidence/choice provided to a probit function
    that takes into account the lapse rates
    it also plots the corresponding fit and, if plt_av=True, plots the
    average choice values for different windows of the evidence
    """
    if evidence.shape[0] > 10 and len(np.unique(choice)) == 2:
        # fit
        popt, pcov = curve_fit(probit_lapse_rates,
                               evidence, choice, maxfev=10000)
    # plot averages
        if plt_av:
            av_data = plot_psychoCurves_averages(evidence, choice,
                                                 color=color, figs=figs)
        else:
            av_data = {}
        # plot obtained probit function
        if figs:
            x = np.linspace(np.min(evidence),
                            np.max(evidence), 50)
            # get the y values for the fitting
            y = probit_lapse_rates(x, popt[0], popt[1], popt[2], popt[3])
            if label == '':
                plt.plot(x, y, color=color, lw=0.5)
            else:
                plt.plot(x, y, color=color,  label=label
                         + ' b: ' + str(round(popt[1], 3)), lw=0.5)
                plt.legend(loc="lower right")
            plot_dashed_lines(-np.max(evidence), np.max(evidence))
    else:
        av_data = {}
        popt = [0, 0, 0, 0]
        pcov = 0
        print('not enough data!')
    return popt, pcov, av_data


def plot_psychoCurves_averages(x_values, y_values,
                               color=(0, 0, 0), figs=False):
    """
    plots average values of y_values for 10 (num_values) different windows
    in x_values
    """
    num_values = 10
    conf = 0.95
    x, step = np.linspace(np.min(x_values), np.max(x_values),
                          num_values, retstep=True)
    curve_mean = []
    curve_std = []
    # compute mean for each window
    for ind_x in range(num_values-1):
        inds = (x_values >= x[ind_x])*(x_values < x[ind_x+1])
        mean = np.mean(y_values[inds])
        curve_mean.append(mean)
        curve_std.append(conf*np.sqrt(mean*(1-mean)/np.sum(inds)))

    if figs:
        # make color weaker
        # np.max(np.concatenate((color, [1, 1, 1]), axis=0), axis=0)
        color_w = np.array(color) + 0.5
        color_w[color_w > 1] = 1
        # plot
        plt.errorbar(x[:-1] + step / 2, curve_mean, curve_std,
                     color=color_w, marker='+', linestyle='')

    # put values in a dictionary
    av_data = {'mean': curve_mean, 'std': curve_std, 'x': x[:-1]+step/2}
    return av_data


def block_stats_perf(performance, blk_dur=1, stps=10**10):
    """
    gets performance for different periods of the block
    """

    # get periods within block to filter the data
    _, aux = np.mgrid[0:performance.shape[0], 0:performance.shape[1]]
    perds = np.floor((aux % blk_dur) / stps)
    perds_vals = np.unique(perds)

    perf_mat = np.zeros((len(perds_vals),))
    for ind_per in range(len(perds_vals)):
        # filter data
        inds = (perds == ind_per)
        performance_block = performance[inds]

        perf_mat[ind_per] = np.mean(performance_block)

    return perf_mat


def block_stats_bias(evidence, stims_pos, performance, action,
                     blk_dur=1, stps=10**10, folder='',
                     name='', plt_av=True, figs=False):
    """
    gets bias stats for different periods of the block
    """
    # get periods within block to filter the data
    _, aux = np.mgrid[0:evidence.shape[0], 0:evidence.shape[1]]
    perds = np.floor((aux % blk_dur) / stps)
    perds_vals = np.unique(perds)

    colors = [[1, 0, 0], [0, 0, 1]]

    bias_mat = np.zeros((len(perds_vals), 2))
    for ind_per in range(len(perds_vals)):
        if figs:
            rows = 2
            cols = 2
            f = plt.figure(figsize=(8, 8), dpi=250)
            matplotlib.rcParams.update({'font.size': 6})
            plt.subplots_adjust(left=left, bottom=bottom, right=right,
                                top=top, wspace=wspace, hspace=hspace)
        else:
            rows = 0
            cols = 0
        # filter data
        inds = (perds == ind_per)
        evidence_block = evidence[inds]
        performance_block = performance[inds]
        action_block = action[inds]
        data = get_psyCho_curves_data(performance_block,
                                      evidence_block, action_block,
                                      0,
                                      rows, cols, figs, colors[0],
                                      plt_av, data={})
        bias_mat[ind_per, 0] = data['popt_repProb_hits_0'][1]
        bias_mat[ind_per, 1] = data['popt_repProb_fails_0'][1]

        if folder != '':
            np.savez(folder + '/figures/psychCurv_stats' +
                     name + '_' + str(ind_per) + '.npz', **data)
            if figs:
                f.savefig(folder+'/figures/psychCurv_' + name + '_' +
                          str(ind_per) + '.svg', dpi=600, bbox_inches='tight')
                plt.close(f)

    return bias_mat


def blk_stats_act(net_st, rep_prob, blk_dur=1, stps=10**10):
    """
    gets average neural activity for different periods of the block
    """

    # get periods within block to filter the data
    _, aux = np.mgrid[0:rep_prob.shape[0], 0:rep_prob.shape[1]]
    perds = np.floor((aux % blk_dur) / stps)
    perds_vals = np.unique(perds)

    # repeating probs. values
    probs_vals = np.unique(rep_prob)
    assert len(probs_vals) <= 2

    act_mat = np.zeros((len(perds_vals), len(probs_vals), net_st.shape[1]))
    for ind_per in range(len(perds_vals)):
        for ind_blk in range(len(probs_vals)):
            # filter data
            inds = (rep_prob == probs_vals[ind_blk]) *\
                (perds == ind_per)

            net_st_block = net_st[inds[0, :], :]
            act_mat[ind_per, ind_blk, :] = np.mean(net_st_block, axis=0)

    return act_mat


def plot_dashed_lines(minimo, maximo):
    plt.plot([0, 0], [0, 1], '--k', lw=0.2)
    plt.plot([minimo, maximo], [0.5, 0.5], '--k', lw=0.2)


def condition_on_prev_hist(stim_pos, stps, rep_alt):
    aux = np.random.choice([0, 1])
    stim_pos = np.concatenate((np.array(aux).reshape(1,), stim_pos))

    repeat = np.diff(stim_pos)
    mask = repeat == 0
    shifted = repeat

    for ind_prev in range(stps-1):
        shifted = np.concatenate(
                    (np.array(np.random.choice([0, 1])).reshape(1,),
                     shifted[:-1]))
        if rep_alt == 'rep':
            mask = mask * (shifted == 0)
        elif rep_alt == 'alt':
            mask = mask * (shifted != 0)
    return mask


def plot_trials(evidence, trial_duration, performance, stim_position,
                num_trials, rewards_trials, rewards):
    """
    plot several consecutive trials, showing the average of the 2 stimuli,
    the reward signal, the performance of the network
    """
    plt.figure(figsize=(8, 2), dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)
    steps_counter = 0
    for ind_tr in range(num_trials):
        # if you plot the reward, this should be 2 so the 2 plots are separated
        dc = 0
        # plot stim1
        index = np.arange(steps_counter, steps_counter+trial_duration[ind_tr])
        color1 = (1-stim_position[ind_tr], 0, stim_position[ind_tr])
        color2 = (stim_position[ind_tr], 0, 1-stim_position[ind_tr])
        plt.plot(index, dc+evidence[ind_tr]*np.ones((trial_duration[ind_tr]),),
                 color=color1)
        # plot evidence
        plt.plot(index, dc+np.ones((trial_duration[ind_tr]),),
                 color=color2)
        if ind_tr < num_trials:
            plt.plot([steps_counter, steps_counter], [dc, 1+dc], '--',
                     color=(.6, .6, .6), lw=0.5)
        steps_counter += trial_duration[ind_tr]
        if ind_tr < num_trials:
            if performance[ind_tr]:
                plt.plot([steps_counter-1, steps_counter-1], [dc, 1+dc],
                         '--', color=(0, 1, 0), lw=1)
            else:
                plt.plot([steps_counter-1, steps_counter-1], [dc, 1+dc],
                         '--k', lw=1)

        plt.xlabel('time (a.u.)')


def plot_sequence(stims_pos, repeat, num_trials):
    """
    plot the sequence of trials
    """
    plt.figure(figsize=(8, 4), dpi=250)
    matplotlib.rcParams.update({'font.size': 8})
    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)
    # plot 400 first trials (to see the repetition sequences)
    plt.subplot(3, 1, 1)
    plt.imshow(stims_pos.reshape(1, stims_pos.shape[0]), aspect='auto')
    plt.ylabel('sequence')
    plt.yticks([])
    plt.subplot(3, 1, 2)
    plt.imshow(stims_pos[0:num_trials].reshape(1, num_trials), aspect='auto')
    plt.ylabel('sequence (zoom)')
    plt.yticks([])
    plt.subplot(3, 1, 3)
    plt.plot(repeat[0:num_trials])
    plt.xlim(0, num_trials)
    plt.xlabel('trials')
    plt.ylabel('repeat')
    plt.yticks([0, 1])


def plot_fractions(lims):
    """
    plot dashed lines for 0.25, 0.5 and 0.75
    """
    plt.plot(lims, [0.25, 0.25], '--k', lw=0.25)
    plt.plot(lims, [0.5, 0.5], '--k', lw=0.25)
    plt.plot(lims, [0.75, 0.75], '--k', lw=0.25)
    plt.xlim(lims[0], lims[1])


def roc_analysis(X, y):
    """
    compute the roc curve
    """
    fpr, tpr, _ = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def probit_lapse_rates(x, beta, alpha, piL, piR):
    piR = 0
    piL = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def probit(x, beta, alpha):
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def ideal_observer(evidence, stim_position, alpha=0, tau=2, window=100):
    """
    computes the ideal observer choice taking into account
    several (window) previous trials
    """
    evidence /= np.max(evidence.flatten())
    # ideal observer choice
    if alpha != 0:
        kernel = np.arange(window)
        kernel = np.exp(-kernel/tau)[::-1]
        kernel /= np.sum(kernel)
        final_choice = np.zeros_like(stim_position)
        for ind_w in range(stim_position.shape[0]):
            trial_history = -2*(np.convolve(stim_position[ind_w, :],
                                            kernel, mode='same') - 0.5)
            final_choice[ind_w, :] = ((1-alpha)*evidence[ind_w, :] +
                                      alpha*trial_history) < 0
    else:
        final_choice = evidence < 0

    return final_choice


def pca_3d(net_st, cond_mat, values, ind, title):
    plt.figure()
    num_neurons = int(net_st.shape[1]/12)
    print(num_neurons)
    ax = plt.axes(projection='3d')
    X = np.array(net_st[:, num_neurons*ind:num_neurons*(ind+1)])
    pca = PCA(n_components=3)
    aux = pca.fit_transform(X)
    inds = cond_mat[ind, :] == values[0]
    ax.scatter3D(aux[inds, 0], aux[inds, 1], aux[inds, 2], color='r')
    inds = cond_mat[ind, :] == values[1]
    ax.scatter3D(aux[inds, 0], aux[inds, 1], aux[inds, 2], color='b')
    plt.title(title)

    plt.figure()
    num_neurons = int(net_st.shape[1]/12)
    ax = plt.axes(projection='3d')
    X = np.array(net_st[:, num_neurons*ind:num_neurons*(ind+1)])
    pca = PCA(n_components=3)
    aux = pca.fit_transform(X)
    inds = cond_mat[ind, :] == values[0]
    ax.scatter3D(aux[inds, 1], aux[inds, 0], aux[inds, 2], color='r')
    inds = cond_mat[ind, :] == values[1]
    ax.scatter3D(aux[inds, 1], aux[inds, 0], aux[inds, 2], color='b')
    plt.title(title)

    plt.figure()
    num_neurons = int(net_st.shape[1]/12)
    ax = plt.axes(projection='3d')
    X = np.array(net_st[:, num_neurons*ind:num_neurons*(ind+1)])
    pca = PCA(n_components=3)
    aux = pca.fit_transform(X)
    inds = cond_mat[ind, :] == values[0]
    ax.scatter3D(aux[inds, 2], aux[inds, 0], aux[inds, 1], color='r')
    inds = cond_mat[ind, :] == values[1]
    ax.scatter3D(aux[inds, 2], aux[inds, 0], aux[inds, 1], color='b')
    plt.title(title)


def plot_cond(pcs, mat_cond, values, name, st, mask, num_samples):
    mat_cond_aux = mat_cond[0, st:].copy()
    mat_cond_aux = mat_cond_aux[mask]

    pcs = pcs[-num_samples:, :]
    mat_cond_aux = mat_cond_aux[-num_samples:]

    inds = mat_cond_aux == values[0]
    plt.scatter(pcs[inds, 0], pcs[inds, 1], s=80, facecolors='none',
                edgecolors='r', label=name+' '+str(values[0]))
    inds = mat_cond_aux == values[1]
    plt.scatter(pcs[inds, 0], pcs[inds, 1], s=80, facecolors='none',
                edgecolors='b', label=name+' '+str(values[1]))
    plt.title(name)
    plt.legend()


def plot_pcs(pcs, mat_cond, values, name, st, mask, panel, grid=[], ind=0):
    colors = 'rbg'
    for ind_pc in range(pcs.shape[1]):
        mat_cond_aux = mat_cond[0, st:].copy()
        mat_cond_aux = mat_cond_aux[mask]
        if len(grid) == 0:
            plt.subplot(5, pcs.shape[1], panel+ind_pc)
        else:
            plt.subplot(grid[0], grid[1], ind+ind_pc)

        for ind_v in range(len(values)):
            inds = mat_cond_aux == values[ind_v]
            aux, bins = np.histogram(pcs[inds, ind_pc], 50)
            aux = aux/np.sum(aux)
            plt.plot(bins[1:], aux, color=colors[ind_v],
                     label=name+' '+str(values[ind_v]), lw=0.5)

        plt.xlabel(name)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        # plt.legend()


def build_block_mat(shape, block_dur):
    # build rep. prob vector
    rp_mat = np.zeros(shape)
    a = np.arange(shape[1])
    b = np.floor(a/block_dur)
    rp_mat[:, b % 2 == 0] = 1
    return rp_mat


if __name__ == '__main__':
    plt.close('all')
    # exp. duration (num. trials; training consists in several exps)
    exp_dur = 100
    # num steps per trial
    trial_dur = 10
    # rewards given for: stop fixating, keep fixating, correct, wrong
    rewards = (-0.1, 0.0, 1.0, -1.0)
    # number of trials per blocks
    block_dur = 200
    # stimulus evidence
    stim_ev = 0.5
    # prob. of repeating the stimuli in the positions of previous trial
    rep_prob = [0.1, 0.9]
    # model seed
    env_seed = 0
    # folder where data will be saved
    main_folder = '/home/molano/expectations_results'
    exp = main_folder + '/ed_' + str(exp_dur) +\
        '_rp_' +\
        str(utils.list_str(rep_prob)) +\
        '_r_' + str(utils.list_str(rewards)) +\
        '_bd_' + str(block_dur) + '_ev_' +\
        str(stim_ev) + '/' + str(env_seed)
    folder = exp
    num_tr = 260000
    # get experiment params and data
    exp_params = np.load(exp + '/experiment_setup.npz')
    data = np.load(exp + '/trials_stats_0_' + str(num_tr) + '.npz')
    print('num. trials: ' + str(data['evidence'].shape[0]))
    start_per = num_tr - 20000
    # plot psycho. curves
    ev = np.reshape(data['evidence'], (1, data['evidence'].shape[0])).copy()
    perf = np.reshape(data['performance'],
                      (1, data['performance'].shape[0])).copy()
    action = np.reshape(data['action'], (1, data['action'].shape[0])).copy()
    stim_pos = np.reshape(data['stims_position'],
                          (1, data['stims_position'].shape[0])).copy()
    plot_psychometric_curves(ev[:, start_per:], perf[:, start_per:],
                             action[:, start_per:], blk_dur=block_dur,
                             figs=True)
    # plot learning
    ev = np.reshape(data['evidence'], (1, data['evidence'].shape[0])).copy()
    perf = np.reshape(data['performance'],
                      (1, data['performance'].shape[0])).copy()
    action = np.reshape(data['action'], (1, data['action'].shape[0])).copy()
    stim_pos = np.reshape(data['stims_position'],
                          (1, data['stims_position'].shape[0])).copy()
    plot_learning(perf, ev, stim_pos, action, view_fig=True)
    #                  folder='/home/molano/', name='', save_fig=True)

    # test with old data
    start_per = 10000
    exp = '/home/molano/expectations_project/priors/novel_data/' +\
        'main_exp_0208_200_04/trial_duration_10_repeating_prob_0.2_0.8' +\
        '_rewards_-0.1_0.0_1.0_-1.0_block_dur_200_stimEv_0.4_gamma_0.8' +\
        '_num_units_32_update_net_step_5_network_ugru_11/trains/train_0/' +\
        'trials_stats_50000.npz'
    data = np.load(exp)
    ev = np.reshape(data['evidence'], (1, data['evidence'].shape[0])).copy()
    perf = np.reshape(data['performance'],
                      (1, data['performance'].shape[0])).copy()
    action = np.reshape(data['action'], (1, data['action'].shape[0])).copy()
    stim_pos = np.reshape(data['stims_position'],
                          (1, data['stims_position'].shape[0])).copy()
    plot_psychometric_curves(ev[:, start_per:], perf[:, start_per:],
                             action[:, start_per:], blk_dur=block_dur,
                             figs=True)
