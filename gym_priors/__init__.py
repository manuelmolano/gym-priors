from gym.envs.registration import register
register(
    id='priors-v0',
    entry_point='gym_priors.envs:PriorsEnv')