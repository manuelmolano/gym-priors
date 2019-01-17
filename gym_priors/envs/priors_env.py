import os
import gym
import numpy as np
from gym import spaces
import sys
# from gym.utils import seeding


class PriorsEnv(gym.Env):
    metadata = {}

    def __init__(self, exp_dur=100, trial_dur=10,
                 rep_prob=None, rewards=(0.1, -0.1, 1.0, -1.0), env_seed='0',
                 block_dur=200, stim_ev=0.5, folder=''):
        print('init environment!')
        # exp. duration (training will consist in several experiments)
        self.exp_dur = exp_dur
        # num steps per trial
        self.trial_dur = trial_dur
        # rewards given for: stop fixating, keep fixating, correct, wrong
        self.rewards = rewards
        # number of trials per blocks
        self.block_dur = block_dur
        # stimulus evidence: one stimulus is always N(1,1), the mean of
        # the other is drawn from a uniform distrib.=U(stim_ev,1).
        # stim_ev must then be between 0 and 1 and the higher it is
        # the more difficult will be the task
        self.stim_ev = stim_ev
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob
        # model instance
        self.env_seed = env_seed
        # folder to save data
        self.folder = folder

        # num actions
        self.num_actions = 3
        self.action_space = spaces.Discrete(self.num_actions)
        # position of the first stimulus
        self.stms_pos_new_trial = np.random.choice([0, 1])
        # keeps track of the repeating prob of the current block
        self.curr_rep_prob = np.random.choice([0, 1])
        # position of the stimuli
        self.stm_pos_new_trial = 0
        # steps counter
        self.timestep = 0
        # initialize ground truth state [stim1 mean, stim2 mean, fixation])
        # the network has to output the action corresponding to the stim1 mean
        # that will be always 1.0 (I just initialize here at 0 for convinience)
        self.int_st = np.array([0, 0, -1])
        # accumulated evidence
        self.evidence = 0
        # observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(3, ), dtype=np.float32)
        # number of trials
        self.num_tr = 0

        # trial data to save
        # stimulus evidence
        self.ev_mat = []
        # position of stimulus 1
        self.stm_pos = []
        # performance
        self.perf_mat = []
        # summed activity across the trial
        self.action = []

    def update_params(self, exp_dur=None, trial_dur=None, rewards=None,
                      block_dur=None, stim_ev=None, rep_prob=None, folder=None,
                      args=None, seed=0):
        """
        this function should be run after creating an environment to set the
        main parameters.
        There does not seem to be easy way of passing those parameters
        when using the make function of the gym toolbox, so this is a way
        around to set the parameters.
        """
        # add current path to sys.path so as to import utils
        sys.path.append(os.path.dirname(os.path.realpath(__file__)))
        import utils
        print(args)
        # exp. duration (num. trials; training consists in several exps)
        self.exp_dur = exp_dur or args.exp_dur or self.exp_dur
        # num steps per trial
        self.trial_dur = trial_dur or args.trial_dur or self.trial_dur
        # rewards given for: stop fixating, keep fixating, correct, wrong
        self.rewards = rewards or args.reward or self.rewards
        # number of trials per blocks
        self.block_dur = block_dur or args.block_dur or self.block_dur
        # stimulus evidence
        stim_ev = stim_ev or args.stim_ev or self.stim_ev
        self.stim_ev = np.max([stim_ev, 10e-5])
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob or args.rep_prob or self.rep_prob
        # model seed
        self.env_seed = seed or self.env_seed
        # folder where data will be saved
        aux_folder = folder or args.folder or self.folder
        exp = aux_folder + '/ed_' + str(self.exp_dur) +\
            '_rp_' +\
            str(utils.list_str(self.rep_prob)) +\
            '_r_' + str(utils.list_str(self.rewards)) +\
            '_bd_' + str(self.block_dur) + '_ev_' +\
            str(self.stim_ev) + '/' + str(self.env_seed)
        if not os.path.exists(exp):
            os.makedirs(exp)
        self.folder = exp
        # save environment parameters
        data = {'exp_dur': self.exp_dur, 'rep_prob': self.rep_prob,
                'rewards': self.rewards, 'stim_ev': self.stim_ev,
                'block_dur': self.block_dur,
                'starting_prob': self.rep_prob[self.curr_rep_prob]}

        np.savez(exp + '/experiment_setup.npz', **data)
        print('--------------- Priors experiment ---------------')
        print('Duration of each experiment (in trials): ' +
              str(self.exp_dur))
        print('Duration of each trial (in steps): ' + str(self.trial_dur))
        print('Rewards: ' + str(self.rewards))
        print('Duration of each block (in trials): ' + str(self.block_dur))
        print('Repeating probabilities of each block: ' + str(self.rep_prob))
        print('Stim evidence: ' + str(self.stim_ev))
        print('Saving folder: ' + str(self.folder))
        print('--------------- ----------------- ---------------')

    def step(self, action):
        """
        receives an action and returns a reward, a state and flag variables
        that indicate whether to start a new trial and whether to update
        the network
        """
        new_trial = True
        correct = False
        done = False
        # decide which reward and state (new_trial, correct) we are in
        if self.timestep < self.trial_dur:
            if (self.int_st[action] != -1).all():
                reward = self.rewards[0]
            else:
                # don't abort the trial even if the network stops fixating
                reward = self.rewards[1]

            new_trial = False

        else:
            if (self.int_st[action] == 1.0).all():
                reward = self.rewards[2]
                correct = True
            else:
                reward = self.rewards[3]

        info = {'perf': correct, 'ev': self.evidence}

        if new_trial:
            # keep main variables of the trial
            self.stm_pos.append(self.stms_pos_new_trial)
            self.perf_mat.append(correct)
            self.action.append(action)
            self.ev_mat.append(self.evidence)
            new_st = self.new_trial()
            # check if it is time to update the network
            done = ((self.num_tr-1) % self.exp_dur == 0) and (self.num_tr != 1)
            # check if it is time to save the trial-to-trial data
            if self.num_tr % 10000 == 0:
                self.save_trials_data()
                self.output_stats()
        else:
            new_st = self.get_state()

        return new_st, reward, done, info

    def get_state(self):
        """
        Outputs a new observation using stim 1 and 2 means.
        It also outputs a fixation signal that is always -1 except at the
        end of the trial that is 0
        """
        self.timestep += 1
        # if still in the integration period present a new observation
        if self.timestep < self.trial_dur:
            self.state = [np.random.normal(self.int_st[0]),
                          np.random.normal(self.int_st[1]), -1]
        else:
            self.state = [0, 0, 0]

        # update evidence
        self.evidence += self.state[0]-self.state[1]

        return np.reshape(self.state, (3, ))

    def new_trial(self):
        """
        this function creates a new trial, deciding the amount of coherence
        (through the mean of stim 2) and the position of stim 1. Once it has
        done this it calls get_state to get the first observation of the trial
        """
        self.num_tr += 1
        self.timestep = 0
        self.evidence = 0
        # this are the means of the two stimuli
        stim1 = 1.0
        stim2 = np.random.uniform(1-self.stim_ev, 1)
        assert stim2 != 1.0
        self.choices = [stim1, stim2]

        # decide the position of the stims
        # if the block is finished update the prob of repeating
        if self.num_tr % self.block_dur == 0:
            self.curr_rep_prob = int(not self.curr_rep_prob)

        # flip a coin
        repeat = np.random.uniform() < self.rep_prob[self.curr_rep_prob]
        if not repeat:
            self.stms_pos_new_trial = not(self.stms_pos_new_trial)

        aux = [self.choices[x] for x in [int(self.stms_pos_new_trial),
               int(not self.stms_pos_new_trial)]]

        self.int_st = np.concatenate((aux, np.array([-1])))

        # get state
        s = self.get_state()

        return s

    def save_trials_data(self):
        """
        save trial-to-trial data for:
        evidence, stim postion, action taken and outcome
        """
        # Periodically save model trials statistics.
        data = {'stims_position': self.stm_pos,
                'action': self.action,
                'performance': self.perf_mat,
                'evidence': self.ev_mat}
        np.savez(self.folder + '/trials_stats_' +
                 str(self.env_seed) + '_' + str(self.num_tr) + '.npz', **data)

    def reset(self):
        return self.new_trial()

    def output_stats(self):
        """
        plot temporary learning and bias curves
        """
        # add current path to sys.path so as to import analyses_priors
        sys.path.append(os.path.dirname(os.path.realpath(__file__)))
        import analyses_priors as ap
        aux_shape = (1, len(self.ev_mat))
        # plot psycho. curves
        per = 20000
        ev = np.reshape(self.ev_mat, aux_shape).copy()
        ev = ev[np.max([0, len(ev)-per]):]
        perf = np.reshape(self.perf_mat,
                          aux_shape).copy()
        perf = perf[np.max([0, len(perf)-per]):]
        action = np.reshape(self.action, aux_shape).copy()
        action = action[np.max([0, len(action)-per]):]
        stim_pos = np.reshape(self.stm_pos,
                              aux_shape).copy()
        stim_pos = stim_pos[np.max([0, len(stim_pos)-per]):]
        ap.plot_psychometric_curves(ev, perf, action, blk_dur=self.block_dur,
                                    figs=True, folder=self.folder,
                                    name='psycho_'+str(self.num_tr))
        # plot learning
        ev = np.reshape(self.ev_mat, aux_shape).copy()
        perf = np.reshape(self.perf_mat,
                          aux_shape).copy()
        action = np.reshape(self.action, aux_shape).copy()
        stim_pos = np.reshape(self.stm_pos,
                              aux_shape).copy()
        ap.plot_learning(perf, ev, stim_pos, action, folder=self.folder,
                         name='', save_fig=True, view_fig=True)

    def render():
        pass
