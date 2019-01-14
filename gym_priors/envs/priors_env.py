import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
# from gym.utils import seeding


class PriorsEnv(gym.Env):
    metadata = {}

    def __init__(self, exp_dur=100, trial_duration=10,
                 repeating_prob=(0.2, 0.7), rewards=(-0.1, 0.0, 1.0, -1.0),
                 block_dur=200, stim_evidence=0.5):
        print('init environment!')
        # exp. duration (training will consist in several experiments)
        self.exp_dur = exp_dur
        # num steps per trial
        self.trial_duration = trial_duration
        # rewards given for: stop fixating, keep fixating, correct, wrong
        self.rewards = rewards
        # number of trials per blocks
        self.block_dur = block_dur
        # stimulus evidence: one stimulus is always N(1,1), the mean of
        # the other is drawn from a uniform distrib.=U(stim_ev,1).
        # stim_evidence must then be between 0 and 1 and the higher it is
        # the more difficult will be the task
        self.stim_evidence = np.max([stim_evidence, 10e-5])
        # prob. of repeating the stimuli in the positions of previous trial
        self.repeating_prob = repeating_prob

        # num actions
        self.num_actions = 3
        self.action_space = spaces.Discrete(self.num_actions)
        # keeps track of the repeating prob of the current block
        self.curr_rep_prob = np.random.choice([0, 1])
        # position of the stimuli
        self.stms_pos_new_trial = 0
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
        self.evidence_mat = []
        # position of stimulus 1
        self.stms_pos = []
        # performance
        self.perf_mat = []
        # current repeating probability
        self.rep_prob = []
        # summed activity across the trial
        self.action = []

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
        if self.timestep < self.trial_duration:
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
            self.perf_mat.append(correct)
            self.action.append(action)
            self.evidence_mat.append(self.evidence)
            new_st = self.new_trial()
            # check if it is time to update the network
            done = ((self.num_tr-1) % self.exp_dur == 0) and (self.num_tr != 1)
            if self.num_tr % 10000 == 0:
                self.save_trials_data()
            self.render()
        else:
            new_st = self.get_state()

        return new_st, reward, done, info

    def get_state(self):
        self.timestep += 1  # this was previously in pullArm
        if self.timestep < self.trial_duration:
            self.state = [np.random.normal(self.int_st[0], scale=1),
                          np.random.normal(
                                           self.int_st[1],
                                           scale=1), -1]
        else:
            self.state = [0, 0, 0]

        self.evidence += self.state[0]-self.state[1]
        self.state = np.reshape(self.state, [1, self.num_actions, 1])

        return np.reshape(self.state, (3, ))

    def new_trial(self):
        self.num_tr += 1
        self.timestep = 0
        self.evidence = 0
        # this are the means of the two stimuli
        stim1 = 1.0
        stim2 = np.random.uniform(1-self.stim_evidence, 1)
        assert stim2 != 1.0
        self.choices = [stim1, stim2]

        # decide the position of the stims
        # if the block is finished update the prob of repeating
        if self.num_tr % self.block_dur == 0:
            self.curr_rep_prob = int(not self.curr_rep_prob)

        # flip a coin
        repeat = np.random.uniform() < self.repeating_prob[self.curr_rep_prob]
        if not repeat:
            self.stms_pos_new_trial = not(self.stms_pos_new_trial)

        aux = [self.choices[x] for x in [int(self.stms_pos_new_trial),
               int(not self.stms_pos_new_trial)]]

        self.int_st = np.concatenate((aux, np.array([-1])))

        # keep position
        self.stms_pos.append(self.stms_pos_new_trial)

        # get state
        s = self.get_state()

        return s

    def save_trials_data(self):
        # Periodically save model trials statistics.
        data = {'stims_position': self.stms_pos,
                'action': self.action,
                'performance': self.perf_mat,
                'evidence': self.evidence_mat}
        np.savez(self.folder +
                 '/trials_stats_' + str(self.num_tr) + '.npz', **data)

    def reset(self):
        return self.new_trial()

    def render(self, mode='', close=False):
        conv_w = 20
        plt.plot(np.convolve(self.perf_mat, np.ones((conv_w, )) / conv_w))
        plt.show()
        print('plot!')
