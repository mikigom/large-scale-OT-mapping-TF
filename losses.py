import tensorflow as tf
from costs import cost_l2

__eps__ = .1


def F_entropy(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return -eps*tf.exp((ux + vy - c(x, y))/eps)


def H_entropy(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return tf.exp((ux + vy - c(x, y))/eps)


def reg_ot_dual(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return ux + vy + F_entropy(ux, vy, x, y, c, eps)


def primer_dual(fx, ux, vy, x, y, c=cost_l2, eps=__eps__):
    # f(x) in Y
    return c(y, fx)*H_entropy(ux, vy, x, y, c, eps)
