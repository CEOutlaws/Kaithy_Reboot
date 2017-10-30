import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import copy


def main():
    '''
    AI Self-training program
    '''
    env = gym.make('Gomoku9x9-v0')
    observation = env.reset()

    while True:
        action = env.action_space.sample()  # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
