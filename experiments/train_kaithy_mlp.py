import sys
sys.path.append('..')

import numpy as np

import adversarial_gym as gym
from baselines import deepq


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    env = gym.make('Gomoku9x9-training-camp-v0', None)
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        flatten_obs=True,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=True
    )
    print("Saving model to kaithy_mlp_model.pkl")
    act.save("kaithy_mlp_model.pkl")


if __name__ == '__main__':
    main()
