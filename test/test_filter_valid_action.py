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
    q_values_random = np.array([
        # (1.2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        # (2.5, -2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (1, -3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    ])
    obs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
    invalid_mask_ph = tf.placeholder(
        dtype=tf.float32, shape=[None] + list([env.action_space.n]))
    # Compute q_values
    q_values = tf.constant(q_values_random, dtype=tf.float32)
    # Get min
    q_values_worst = tf.reduce_min(q_values, axis=1)
    # https://stackoverflow.com/questions/1550130/cloning-row-or-column-vectors
    # >>> tile(array([[1,2,3]]).transpose(), (1, 3))
    # array([[1, 1, 1],
    #        [2, 2, 2],
    #        [3, 3, 3]])
    # Assign q_value = min q_value if invalid action,before find action that give max q_value
    # q_values_worst_mask = tf.tile(tf.transpose(
    #     [q_values_worst]), [1, env.action_space.sn])
    # q_values = tf.where(invalid_mask_ph, q_values_worst_mask - one_mask, q_values)
    # one_mask = tf.fill(q_values.shape, 1.0)
    # q_values = tf.where(invalid_mask_ph, q_values - one_mask, q_values)
    q_values = invalid_mask_ph * (q_values_worst - 1.0) + \
        (1.0 - invalid_mask_ph) * q_values
    deterministic_actions = tf.argmax(q_values, axis=1)

    sess = tf.Session()
    for i in range(2):
        observation = env.reset()
        done = None

        while not done:
            action = env.action_space.sample()  # sample without replacement
            observation, reward, done, info = env.step(action)
            env.render()

            print(sess.run(q_values, feed_dict={
                  obs_ph: observation[None],
                  invalid_mask_ph: env.action_space.invalid_mask[None]}))

        env.swap_role()
        print("\n----SWAP----\n")


if __name__ == "__main__":
    main()
