from gym.envs.registration import register

register(
    id='apl-v0',
    entry_point='gym_apl.envs:AplEnv',
)

register(
    id='apl-v1',
    entry_point='gym_apl.envs:AplDropEnv',
)
