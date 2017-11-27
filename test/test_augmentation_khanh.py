import sys
sys.path.append('..')

import tensorflow as tf


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
    if (k == 1):
        rot_pos = pos_2D[0]+(board_size-1 - pos_2D[1] ) *board_size
    # rot180
    if (k == 2):
        rot_pos = (board_size - 1 - pos_2D[0]) * \
            board_size + (board_size - 1 - pos_2D[1])
    # rot270
    if (k == 3):
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
    if (k == 0):
        flip_rot = pos_2D[0] * board_size + -pos_2D[1] + board_size - 1
    # flip and rot 90
    if (k == 1):
        flip_rot = pos_2D[1] * board_size + pos_2D[0]
    # flip and rot 180
    if (k == 2):
        flip_rot = (-pos_2D[0] + board_size - 1) * \
            board_size + pos_2D[1]
    # flip and rot 270
    if (k == 3):
        flip_rot = (-pos_2D[1] + board_size - 1) * \
            board_size + -pos_2D[0] + board_size - 1
    return flip_rot


import adversarial_gym as gym
import numpy as np
import copy


def main():
    board_size = 5
    # Build graph
    act_t_ph = tf.placeholder(tf.int32, [None], name="action")
    batch_size = tf.shape(act_t_ph)[0]
    act_flip_rot = flip_action(board_size, act_t_ph, 1)
    act_rot = tf.zeros_like(tf.int32, [batch_size], name="action_rot")
    act_rot[0:batch_size] = rotate_action(board_size, act_t_ph, 1)

    # Run graph
    sess = tf.Session()
    act_result = sess.run(act_flip_rot, feed_dict={
                          act_t_ph: [1, 2, 3, 4, 5, 6]})
    print(act_result)

    act_result = sess.run(act_rot, feed_dict={act_t_ph: [1, 2, 3, 4, 5, 6]})
    print(act_result)


if __name__ == "__main__":
    main()
