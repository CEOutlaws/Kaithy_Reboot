import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import os
from math import floor, ceil, pi


def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


def rotate(X_imgs):
    rotated_imgs = rotate_images(X_imgs)
    print(rotated_imgs.shape)

    # obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
    # act_t_ph = tf.placeholder(tf.int32, [None], name="action")
    # rew_t_ph = tf.placeholder(U.data_type, [None], name="reward")
    # obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))


def opponent_policy(curr_state, prev_state, prev_action):
    '''
    Define policy for opponent here
    '''
    return gym.gym_gomoku.envs.util.make_beginner_policy(np.random)(curr_state, prev_state, prev_action)


def main():
    '''
    AI Self-training program
    '''
    deterministic_filter = True
    random_filter = True
    env = gym.make('Gomoku5x5-training-camp-v0', opponent_policy)

    num_actions = env.action_space.n

    # obs_ph = tf.placeholder(
    #     dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
    # q_values = layers.fully_connected(layers.flatten(obs_ph), num_actions)
    def make_obs_ph(name):
        obs_shape = env.observation_space.shape

        if flatten_obs:
            flattened_env_shape = 1
            for dim_size in env.observation_space.shape:
                flattened_env_shape *= dim_size
            obs_shape = (flattened_env_shape,)

        return U.BatchInput(obs_shape, name=name)

    #  Create batch aumentation for obs ------------------------------------------

    obs_t_input = tf.placeholder(
        dtype=tf.float32, shape=list(env.observation_space.shape))

    list_obs = []
    list_obs.append(obs_t_input)
    for i in range(0, 8):
        if (i > 0 and i < 4):
            list_obs.append(tf.image.rot90(obs_t_input, k=i))
        if (i == 4):
            list_obs.append(tf.image.flip_left_right(obs_t_input))
        if (i > 4 and i < 8):
            list_obs.append(tf.image.rot90(obs_t_input, k=(i - 4)))

    obs_ph = tf.stack(list_obs)

    # end create augmentation----------------------------------------

    for i in range(2):
        observation = env.reset()
        done = None
        while not done:

            # run to get action from AI
            actions = sess.run(output_actions, feed_dict={
                obs_t_input: observation})

            # Get first valid action
            action = actions[0]

            # Rotate this action
            for i in range(1, 8):
                if (i < 4):
                    actions[i] = rotate_action(observation.shape[0], action, i)
                else:
                    actions[i] = flip_action(
                        observation.shape[0], action, (i - 4))

            # END create actions --------------------------------

            observation, reward, done, info = env.step(action)

            angle = 1
            flip_action(observation.shape[0], action, angle)
            print(action, rotate_action(observation.shape[0], action, angle))
            # exit(0)
            #  observation flip and rotate
            print(observation[:, :, 1], env.observation_space.shape[0:2])
            # exit(0)
            obs_temp_ph = tf.placeholder(
                dtype=tf.int32, shape=(env.observation_space.shape))
            k = tf.placeholder(tf.int32)
            tf_img = tf.image.rot90(obs_temp_ph, k=k)
            rotated_img = sess.run(
                tf_img, feed_dict={obs_temp_ph: observation, k: angle})
            print(rotated_img[:, :, 1])
            exit(0)

            # end obser
            env.render()
            observations.append(observation)

        print(reward)
        env.swap_role()
        print("\n----SWAP----\n")


if __name__ == "__main__":
    main()

import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np


def main():
    board_size = 5
    # Build graph
    act_t_ph = tf.placeholder(tf.int32, [None], name="action")
    batch_size = tf.shape(act_t_ph)[0]
    act_t_aug = tf.concat((
        act_t_ph[0: batch_size // 8],
        rotate_action(
            board_size, act_t_ph[batch_size // 8: batch_size // 4], 1),
        rotate_action(
            board_size, act_t_ph[batch_size // 4: batch_size // 8 * 3], 2),
        rotate_action(
            board_size, act_t_ph[batch_size // 8 * 3: batch_size // 2], 3),
        flip_action(
            board_size, act_t_ph[batch_size // 2: batch_size // 8 * 5], 0),
        flip_action(
            board_size, act_t_ph[batch_size // 8 * 5: batch_size // 8 * 6], 1),
        flip_action(
            board_size, act_t_ph[batch_size // 8 * 6: batch_size // 8 * 7], 2),
        flip_action(
            board_size, act_t_ph[batch_size // 8 * 7: batch_size], 3),
    ), axis=0)

    # Run graph
    sess = tf.Session()

    act_result = sess.run(act_t_ph, feed_dict={
                          act_t_ph: np.arange(board_size * board_size)})
    print(act_result)

    act_result = sess.run(act_t_aug, feed_dict={
                          act_t_ph: np.arange(board_size * board_size)})
    print(act_result)

    assert np.array_equal(act_result, [0,  1,  2,  5,  0, 21, 18, 17, 16,
                                       23,  2,  7, 12, 11, 10,  3,  8, 13,  8,  9,  0, 15, 10,  5,  0])


if __name__ == "__main__":
    main()
