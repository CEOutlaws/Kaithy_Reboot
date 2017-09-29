import gym
import gym_gomoku
import numpy as np


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    # default 'beginner' level opponent policy
    env = gym.make('Gomoku9x9-training-camp-v0')

    # A trick
    # Reset env and attach defined opponent policy to env
    env.step(opponent_policy)
    while True:
        action = env.action_space.sample()  # sample without replacement
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == "__main__":
    main()
