import sys
sys.path.append('..')

import numpy as np

import adversarial_gym as gym
from baselines import deepq
import config as cf


def val_opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent to validate model here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    env = gym.make('Gomoku5x5-training-camp-v0')
    val_env = gym.make('Gomoku5x5-training-camp-v0', val_opponent_policy)
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.cnn_to_mlp(
        convs=[(256, 3, 1), (256, 3, 1), (256, 3, 1), (256, 3, 1),
               (256, 3, 1), (256, 3, 1), (256, 3, 1), (256, 3, 1)],
        hiddens=[256]
    )
    act = deepq.learn(
        env=env,
        val_env=val_env,
        q_func=model,
        max_timesteps=int(sys.argv[1]),
        **cf.gomoku
    )
    print("Saving model to kaithy_cnn_to_mlp_5_model.pkl")
    act.save("kaithy_cnn_to_mlp_5_model.pkl")


if __name__ == '__main__':
    main()
