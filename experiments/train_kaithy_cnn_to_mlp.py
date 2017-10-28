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
    env = gym.make('Gomoku9x9-training-camp-v0', opponent_policy)
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256]
    )
    act = deepq.learn(
        env,
        q_func=model,
        adversarial=True,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    print("Saving model to kaithy_mlp_model.pkl")
    act.save("kaithy_mlp_model.pkl")


if __name__ == '__main__':
    main()
