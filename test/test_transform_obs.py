import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow import image
import numpy as np


def main():
    board_size = 5
    # Build graph
    obs_t_input = tf.placeholder(tf.int32, [None], name="action")
    batch_size = tf.shape(obs_t_input)[0]
    list_obs = []
    list_obs.append(obs_t_input)
    for i in range(0, 8):
        if (i > 0 and i < 4):
            list_obs.append(tf.image.rot90(obs_t_input, k=i))
        if (i == 4):
            list_obs.append(tf.image.flip_left_right(obs_t_input))
        if (i > 4 and i < 8):
            list_obs.append(tf.image.rot90(obs_t_input, k=(i - 4)))
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

    obs_result = sess.run(obs_t_input, feed_dict={
                          obs_t_input: np.arange(board_size * board_size)})
    print(obs_result)

    obs_result = sess.run(obs_t_aug, feed_dict={
                          obs_t_input: np.arange(board_size * board_size)})
    print(obs_result)

    assert np.array_equal(obs_result, [0,  1,  2,  5,  0, 21, 18, 17, 16,
                                       23,  2,  7, 12, 11, 10,  3,  8, 13,  8,  9,  0, 15, 10,  5,  0])


if __name__ == "__main__":
    main()
