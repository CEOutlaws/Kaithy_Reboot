import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import copy


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    # a = curr_state.board.board_state
    # print(a)
    board = copy.deepcopy(curr_state)
    print(board.board, curr_state.board)
    input("wait")
    for x_pixel in range(0, curr_state.board.board_state.shape[0]):
        for y_pixel in range(0, curr_state.board.board_state.shape[1]):
            # print(curr_state.board.board_state[x_pixel][y_pixel])
            # status_in_pixel = curr_state.board.board_state[x_pixel][y_pixel]
            if (curr_state.board.board_state[x_pixel][y_pixel] == 1):
                curr_state.board.board_state[x_pixel][y_pixel] = 2
            elif curr_state.board.board_state[x_pixel][y_pixel] == 2:
                curr_state.board.board_state[x_pixel][y_pixel] = 1
    # print(a)

    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    env = gym.make('Gomoku9x9-training-camp-v0', opponent_policy)
    env.reset()

    while True:
        action = env.action_space.sample()  # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
