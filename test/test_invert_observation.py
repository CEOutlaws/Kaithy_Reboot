import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import copy


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    # Encode to get observation
    obs = curr_state.board.encode()
    print(obs)
    # Invert observation
    inverted_obs = np.roll(obs, 1, axis=2)
    print(inverted_obs)

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
