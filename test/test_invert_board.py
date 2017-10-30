import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import copy


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    print(curr_state.board.board_state)
    inverted_board = np.copy(curr_state.board.board_state)
    for board_space in np.nditer(inverted_board, op_flags=['readwrite']):
        if board_space == 1:
            # board_space is read only
            # to assign new_value, use board_space[...] = new_value
            board_space[...] = 2
        elif board_space == 2:
            board_space[...] = 1
    print(inverted_board)

    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    env = gym.make('Gomoku9x9-training-camp-v0', opponent_policy)
    env.reset()

    action = env.action_space.sample()  # sample without replacement
    observation, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
