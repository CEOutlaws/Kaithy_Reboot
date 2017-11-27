import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow import image
import numpy as np


def main():
    board_size = 5
    # Build graph
    obs_t_input = tf.placeholder(tf.int32, [None, 5, 5, 3], name="action")
    batch_size = tf.shape(obs_t_input)[0]
    obs_t_aug = tf.concat((
        obs_t_input[0: batch_size // 8],
        image.rot90(
            obs_t_input[batch_size // 8: batch_size // 4], 1),
        image.rot90(
            obs_t_input[batch_size // 4: batch_size // 8 * 3], 2),
        image.rot90(
            obs_t_input[batch_size // 8 * 3: batch_size // 2], 3),
        image.flip_left_right(
            obs_t_input[batch_size // 2: batch_size // 8 * 5]),
        image.rot90(
            image.flip_left_right(obs_t_input[batch_size // 8 * 5: batch_size // 8 * 6]), 1),
        image.rot90(
            image.flip_left_right(obs_t_input[batch_size // 8 * 6: batch_size // 8 * 7]), 2),
        image.rot90(
            image.flip_left_right(obs_t_input[batch_size // 8 * 7: batch_size]), 3),
    ), axis=0)

    # Run graph
    sess = tf.Session()

    obs_to_test = np.zeros((5, 5, 1), np.int32)
    obses_to_test = np.array([obs_to_test] * 16)

    for obs_idx in range(len(obses_to_test)):
        obses_to_test[obs_idx][obs_idx // 5, obs_idx % 5][0] = 1

    obs_result = sess.run(obs_t_input, feed_dict={
                          obs_t_input: obses_to_test})
    print(obs_result)

    obs_result = sess.run(obs_t_aug, feed_dict={
                          obs_t_input: obses_to_test})
    print(obs_result)


if __name__ == "__main__":
    main()
