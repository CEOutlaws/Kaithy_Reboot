import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np


def rotate_action(board_size, pos_1D, k):
    """
    Function rotate board
        :param board_size: size of board 
        :param pos_1D: position in board
        :param k:   1: rotate 90
                    2: rotate 180
                    3: rotate 270
    """
    pos_2D = (pos_1D // board_size, pos_1D % board_size)
    # rot90
    if (k % 4 == 1):
        rot_pos = pos_2D[0] + (board_size - 1 - pos_2D[1]) * board_size
    # rot180
    if (k % 4 == 2):
        rot_pos = (board_size - 1 - pos_2D[0]) * \
            board_size + (board_size - 1 - pos_2D[1])
    # rot270
    if (k % 4 == 3):
        rot_pos = (board_size - 1 - pos_2D[0]) + pos_2D[1] * board_size
    return rot_pos


def flip_action(board_size, pos_1D, k):
    """
    Flip board and rotate
        :param board_size: size of board
        :param pos_1D: position in board
        :param k:   0: only flip
                    1: flip and rotate 90
                    2: flip and rotate 180
                    3: flip and rotate 270
    """
    pos_2D = (pos_1D // board_size, pos_1D % board_size)
    # flip and rot 0
    if (k % 4 == 0):
        flip_rot = pos_2D[0] * board_size + -pos_2D[1] + board_size - 1
    # flip and rot 90
    if (k % 4 == 1):
        flip_rot = pos_2D[1] * board_size + pos_2D[0]
    # flip and rot 180
    if (k % 4 == 2):
        flip_rot = (-pos_2D[0] + board_size - 1) * \
            board_size + pos_2D[1]
    # flip and rot 270
    if (k % 4 == 3):
        flip_rot = (-pos_2D[1] + board_size - 1) * \
            board_size + -pos_2D[0] + board_size - 1
    return flip_rot


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
