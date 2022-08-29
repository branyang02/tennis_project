from gym.envs.registration import register

register(
    id='tennis-v0',
    entry_point='gym_tennis.envs:TennisEnv',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic = True,
)