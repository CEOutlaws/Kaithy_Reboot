import sys
sys.path.append('..')

import numpy as np

import adversarial_gym as gym
from baselines import deepq


def val_opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent to validate model here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    env = gym.make('Gomoku9x9-training-camp-v0', val_opponent_policy)
    act = deepq.load("kaithy_cnn_to_mlp_9_model.pkl")
    # Enabling layer_norm here is import for parameter space noise!

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew
            env.render()
        print("Episode reward", episode_rew)
        input('Hit enter to play next match')


if __name__ == '__main__':
    main()
