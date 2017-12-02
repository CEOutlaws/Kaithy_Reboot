from __future__ import division

import adversarial_gym as gym
# from baselines import deepq
import numpy as np
import models
import simple


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(
        lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make('Gomoku9x9-training-camp-v0', opponent_policy)
    model = models.mlp([64])
    act = simple.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
