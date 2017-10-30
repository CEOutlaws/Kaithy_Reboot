import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import copy


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    env = gym.make('Gomoku9x9-training-camp-v0', opponent_policy)
    observation = env.reset()

    while True:
        action = env.action_space.sample()  # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
