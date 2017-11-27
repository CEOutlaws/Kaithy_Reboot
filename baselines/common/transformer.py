import tensorflow as tf
from tensorflow import image

from . import position as pos


def transform_actions(actions, board_size):
    batch_size = tf.shape(actions)[0]

    return tf.concat((
        actions[0: batch_size // 8],
        pos.rot90(
            board_size, actions[batch_size // 8: batch_size // 4], 1),
        pos.rot90(
            board_size, actions[batch_size // 4: batch_size // 8 * 3], 2),
        pos.rot90(
            board_size, actions[batch_size // 8 * 3: batch_size // 2], 3),
        pos.flip_left_right_rot90(
            board_size, actions[batch_size // 2: batch_size // 8 * 5], 0),
        pos.flip_left_right_rot90(
            board_size, actions[batch_size // 8 * 5: batch_size // 8 * 6], 1),
        pos.flip_left_right_rot90(
            board_size, actions[batch_size // 8 * 6: batch_size // 8 * 7], 2),
        pos.flip_left_right_rot90(
            board_size, actions[batch_size // 8 * 7: batch_size], 3),
    ), axis=0)


def transform_obses(obses):
    batch_size = tf.shape(obses)[0]
    return tf.concat((
        obses[0: batch_size // 8],
        tf.map_fn(lambda obs: image.rot90(obs, 1),
                  obses[batch_size // 8: batch_size // 4]),
        tf.map_fn(lambda obs: image.rot90(obs, 2),
                  obses[batch_size // 4: batch_size // 8 * 3]),
        tf.map_fn(lambda obs: image.rot90(obs, 3),
                  obses[batch_size // 8 * 3: batch_size // 2]),
        tf.map_fn(lambda obs: image.flip_left_right(obs),
                  obses[batch_size // 2: batch_size // 8 * 5]),
        tf.map_fn(lambda obs: image.rot90(image.flip_left_right(obs), 1),
                  obses[batch_size // 8 * 5: batch_size // 8 * 6]),
        tf.map_fn(lambda obs: image.rot90(image.flip_left_right(obs), 2),
                  obses[batch_size // 8 * 6: batch_size // 8 * 7]),
        tf.map_fn(lambda obs: image.rot90(image.flip_left_right(obs), 3),
                  obses[batch_size // 8 * 7: batch_size]),
    ), axis=0)
