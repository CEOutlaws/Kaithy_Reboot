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
    class Opponent(object):
        def __init__(self):
            self.__old_obs = None
            self.__old_action = None
            self.__obs = None

        def policy(self, curr_state, prev_state, prev_action):
            '''
            Define policy for opponent here
            '''
            return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)

    opponent = Opponent()
    env = gym.make('Gomoku5x5-training-camp-v0')
    env.opponent_policy = opponent.policy

    for i in range(2):
        observation = env.reset()
        done = None

        while not done:
            action = env.action_space.sample()  # sample without replacement
            observation, reward, done, info = env.step(action)
            env.render()

        env.swap_role()
        print("\n----SWAP----\n")


if __name__ == "__main__":
    main()
