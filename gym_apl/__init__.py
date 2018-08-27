from gym.envs.registration import register

register(
    id='apl-v0',
    entry_point='gym_apl.env:AplEnv',
)
