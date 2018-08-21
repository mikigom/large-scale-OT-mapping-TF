import tensorflow as tf
from costs import cost_l2

__eps__ = .01


def F_entropy(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return -eps*tf.exp((ux + vy - c(x, y))/eps)


def H_entropy(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return tf.exp((ux + vy - c(x, y))/eps)


def F_l2(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return (-1/(4 * eps)) * (tf.nn.relu(ux + vy - c(x, y)))**2


def H_l2(ux, vy, x, y, c=cost_l2, eps=__eps__):
    return (1 / (2 * eps)) * tf.nn.relu(ux + vy - c(x, y))


def reg_ot_dual(ux, vy, x, y, c=cost_l2, eps=__eps__, reg_type='entropy'):
    if reg_type == 'entropy':
        return ux + vy + F_entropy(ux, vy, x, y, c, eps)
    elif reg_type == 'l2':
        return ux + vy + F_l2(ux, vy, x, y, c, eps)


def primer_dual(fx, ux, vy, x, y, c=cost_l2, eps=__eps__, reg_type='entropy'):
    # f(x) in Y
    if reg_type == 'entropy':
        return c(y, fx) * H_entropy(ux, vy, x, y, c, eps)
    elif reg_type == 'l2':
        return c(y, fx) * H_l2(ux, vy, x, y, c, eps)
