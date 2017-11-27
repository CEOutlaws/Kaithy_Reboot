def rot90(board_size, pos_1d, k):
    """
    Function rotate board
        :param board_size: size of board 
        :param pos_1D: position in board
        :param k:   1: rotate 90
                    2: rotate 180
                    3: rotate 270
    """
    pos_2d = (pos_1d // board_size, pos_1d % board_size)
    # rot90
    if (k % 4 == 1):
        rot_pos = pos_2d[0] + (board_size - 1 - pos_2d[1]) * board_size
    # rot180
    if (k % 4 == 2):
        rot_pos = (board_size - 1 - pos_2d[0]) * \
            board_size + (board_size - 1 - pos_2d[1])
    # rot270
    if (k % 4 == 3):
        rot_pos = (board_size - 1 - pos_2d[0]) + pos_2d[1] * board_size
    return rot_pos


def flip_left_right_rot90(board_size, pos_1d, k):
    """
    Flip board and rotate
        :param board_size: size of board
        :param pos_1D: position in board
        :param k:   0: only flip
                    1: flip and rotate 90
                    2: flip and rotate 180
                    3: flip and rotate 270
    """
    pos_2d = (pos_1d // board_size, pos_1d % board_size)
    # flip and rot 0
    if (k % 4 == 0):
        flip_rot = pos_2d[0] * board_size + -pos_2d[1] + board_size - 1
    # flip and rot 90
    if (k % 4 == 1):
        flip_rot = pos_2d[1] * board_size + pos_2d[0]
    # flip and rot 180
    if (k % 4 == 2):
        flip_rot = (-pos_2d[0] + board_size - 1) * \
            board_size + pos_2d[1]
    # flip and rot 270
    if (k % 4 == 3):
        flip_rot = (-pos_2d[1] + board_size - 1) * \
            board_size + -pos_2d[0] + board_size - 1
    return flip_rot
