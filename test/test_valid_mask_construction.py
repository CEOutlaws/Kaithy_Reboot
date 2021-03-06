import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import tensorflow as tf


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    deterministic_actions_filter = True

    env = gym.make('Gomoku5x5-training-camp-v0', opponent_policy)

    obs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(env.observation_space.shape))

    if deterministic_actions_filter:
        invalid_masks = tf.reduce_sum(obs_ph, axis=3)

    sess = tf.Session()

    observations = []

    for i in range(2):
        observation = env.reset()
        done = None

        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            env.render()
    out = sess.run(invalid_masks, feed_dict={
        obs_ph: observations})
    print(out)
    print(out.shape)


if __name__ == "__main__":
    main()
