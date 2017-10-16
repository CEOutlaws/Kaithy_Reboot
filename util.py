import numpy as np


def invert_board(board):
    inverted_board = np.empty_like(board)

    for space in np.ndindex(board):
        if (space == 1):
            pass

    return inverted_board
