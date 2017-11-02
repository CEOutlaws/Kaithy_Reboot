from gym.envs.registration import register

register(
    id='Gomoku19x19-v0',
    entry_point='adversarial_gym.gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner',
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Gomoku9x9-v0',
    entry_point='adversarial_gym.gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner',  # random policy is the simplest
        'board_size': 9,
    },
    nondeterministic=True,
)

register(
    id='Gomoku5x5-training-camp-v0',
    entry_point='adversarial_gym.gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'player',
        'board_size': 5,
    },
    nondeterministic=True,
)

register(
    id='Gomoku9x9-training-camp-v0',
    entry_point='adversarial_gym.gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'player',
        'board_size': 9,
    },
    nondeterministic=True,
)

register(
    id='Gomoku15x15-training-camp-v0',
    entry_point='adversarial_gym.gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'player',
        'board_size': 15,
    },
    nondeterministic=True,
)
