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
        ret = image
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
