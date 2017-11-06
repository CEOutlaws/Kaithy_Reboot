import tensorflow as tf


def main():
    input = tf.constant([
        [
            [[0, 1, 0], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
        ],
        [
            [[1, 1, 0], [1, 0, 1], [1, 0, 1]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]]
        ],
    ], dtype=tf.float32)
    output = tf.contrib.layers.flatten(
        tf.reduce_sum(input[:, :, :, 1:3], axis=3))
    sess = tf.Session()
    print(sess.run(output))


if __name__ == "__main__":
    main()
