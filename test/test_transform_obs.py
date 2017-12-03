import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np


from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops


def rot90(image, k=1, name=None):
    """
    tf.image.rot90 (Khanh Remix)
    Rotate an image counter-clockwise by 90 degrees.

    Args:
      image: A 3-D tensor of shape `[height, width, channels]`.
      k: A scalar integer. The number of times the image is rotated by 90 degrees.
      name: A name for this operation (optional).

    Returns:
      A rotated 3-D tensor of the same type and shape as `image`.
    """
    with ops.name_scope(name, 'rot90', [image, k]) as scope:
        k = k % 4

        if k == 1:
            ret = array_ops.transpose(array_ops.reverse_v2(image, [1]),
                                      [1, 0, 2], name=scope)
        elif k == 2:
            ret = array_ops.reverse_v2(image, [0, 1], name=scope)

        elif k == 3:
            ret = array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]),
                                       [1], name=scope)

        ret.set_shape([None, None, image.get_shape()[2]])
        return ret


def main():
    # Build graph
    obs_t_input = tf.placeholder(tf.int32, [None, 5, 5, 1], name="action")
    batch_size = tf.shape(obs_t_input)[0]
    obs_t_aug = tf.concat((
        obs_t_input[0: batch_size // 8],
        tf.map_fn(lambda obs: rot90(obs, 1),
                  obs_t_input[batch_size // 8: batch_size // 4]),
        tf.map_fn(lambda obs: rot90(obs, 2),
                  obs_t_input[batch_size // 4: batch_size // 8 * 3]),
        tf.map_fn(lambda obs: rot90(obs, 3),
                  obs_t_input[batch_size // 8 * 3: batch_size // 2]),
        tf.map_fn(lambda obs: tf.image.flip_left_right(obs),
                  obs_t_input[batch_size // 2: batch_size // 8 * 5]),
        tf.map_fn(lambda obs: rot90(tf.image.flip_left_right(obs), 1),
                  obs_t_input[batch_size // 8 * 5: batch_size // 8 * 6]),
        tf.map_fn(lambda obs: rot90(tf.image.flip_left_right(obs), 2),
                  obs_t_input[batch_size // 8 * 6: batch_size // 8 * 7]),
        tf.map_fn(lambda obs: rot90(tf.image.flip_left_right(obs), 3),
                  obs_t_input[batch_size // 8 * 7: batch_size]),
    ), axis=0)

    # Run graph
    sess = tf.Session()

    obs_to_test = np.zeros((5, 5, 1), np.int32)
    obses_to_test = np.array([obs_to_test] * 25)

    for obs_idx in range(len(obses_to_test)):
        obses_to_test[obs_idx][obs_idx // 5, obs_idx % 5][0] = 1

    obses_input = sess.run(obs_t_input, feed_dict={
        obs_t_input: obses_to_test})

    obses_result = sess.run(obs_t_aug, feed_dict={
        obs_t_input: obses_to_test})
    for obs_idx in range(len(obses_result)):
        print(obs_idx)
        print(np.squeeze(obses_input[obs_idx]))
        print(np.squeeze(obses_result[obs_idx]))


if __name__ == "__main__":
    main()
