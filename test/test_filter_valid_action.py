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
    env = gym.make('Gomoku5x5-training-camp-v0', opponent_policy)
    obs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
    invalid_mask_ph = tf.placeholder(
        dtype=tf.bool, shape=[None] + list(env.action_space.invalid_mask.shape))
    sess = tf.Session()
    for i in range(2):
        observation = env.reset()
        done = None

        while not done:
            action = env.action_space.sample()  # sample without replacement
            observation, reward, done, info = env.step(action)
            env.render()

            print(sess.run(obs_ph, feed_dict={obs_ph: observation[None]}))
            print(sess.run(invalid_mask_ph, feed_dict={
                  invalid_mask_ph: env.action_space.invalid_mask[None]}))

        env.swap_role()
        print("\n----SWAP----\n")


if __name__ == "__main__":
    main()
