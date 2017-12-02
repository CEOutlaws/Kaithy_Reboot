import numpy as np

import adversarial_gym as gym
from baselines import deepq


def __val_opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent to validate model here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def train(board_size, max_timesteps):
    """train gomoku AI play board whose size is board_size x board_size.

    Parameters
    ----------
    board_size: int
        Size of board in one dimension, example:
        board_size = 9 --> board have size 9x9
    max_timesteps: int
        Number of training step

    Returns
    -------
    None
    """
    env = gym.make(
        'Gomoku{}x{}-arena-v0'.format(board_size, board_size))
    val_env = gym.make(
        'Gomoku{}x{}-arena-v0'.format(board_size, board_size), __val_opponent_policy)

    # Enabling layer_norm here is import for parameter space noise!
    capility = 64
    num_conv_layer = 8
    conv_layers = [(capility, 3, 1)] * num_conv_layer
    hidden_layers = [capility]

    model = deepq.models.cnn_to_mlp(
        convs=conv_layers,
        hiddens=hidden_layers
    )

    act = deepq.learn(
        env=env,
        val_env=val_env,
        q_func=model,
        max_timesteps=max_timesteps,
        lr=1e-4,
        buffer_size=400000,
        batch_size=512,
        exploration_fraction=0.95,
        # exploration_fraction=0.001,
        exploration_final_eps=0.35,
        train_freq=4,
        val_freq=1000,
        print_freq=100,
        learning_starts=10000,
        # learning_starts=32,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        deterministic_filter=True,
        random_filter=True,
        state_file='kaithy_cnn_to_mlp_{}_model.pkl'.format(board_size),
    )

    print('Saving model to kaithy_cnn_to_mlp_{}_model.pkl'.format(
        board_size))
    act.save('kaithy_cnn_to_mlp_{}_model.pkl'.format(board_size))


def enjoy(board_size):
    """enjoy trained gomoku AI play board whose size is board_size x board_size.

    Parameters
    ----------
    board_size: int
        Size of board in one dimension, example:
        board_size = 9 --> board have size 9x9

    Returns
    -------
    None
    """
    env = gym.make('Gomoku{}x{}-arena-v0'.format(board_size,
                                                 board_size), __val_opponent_policy)
    act = deepq.load("kaithy_cnn_to_mlp_{}_model.pkl".format(
        board_size))
    # Enabling layer_norm here is import for parameter space noise!

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew
            env.render()
        print('Episode reward', episode_rew)
        input('Hit enter to play next match')
        print('Swap color')
        env.swap_role()
