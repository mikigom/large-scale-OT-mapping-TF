import tensorflow as tf


def cost_l2(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)
