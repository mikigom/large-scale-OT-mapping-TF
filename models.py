import tensorflow as tf
slim = tf.contrib.slim

__leaky_relu_alpha__ = 0.2


def __leaky_relu__(x, alpha=__leaky_relu_alpha__, name='Leaky_ReLU'):
    return tf.maximum(x, alpha*x, name=name)


class NN_DUAL(object):
    def __init__(self, input_, vs_name, reuse=False):
        self.input = input_
        self.vs_name = vs_name
        self.reuse = reuse
        self.output = None
        self.var_list = None
        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.vs_name, reuse=self.reuse) as vs:
            with slim.arg_scope([slim.fully_connected],
                                num_outputs=32,
                                activation_fn=__leaky_relu__):
                fc_1 = slim.fully_connected(self.input)
                fc_2 = slim.fully_connected(fc_1)
                fc_3 = slim.fully_connected(fc_2, num_outputs=1, activation_fn=tf.nn.relu)

        self.output = fc_3
        self.var_list = tf.contrib.framework.get_variables(vs)


class NN_MAP(object):
    def __init__(self, input_, vs_name, reuse=False):
        self.input = input_
        self.vs_name = vs_name
        self.reuse = reuse
        self.output = None
        self.var_list = None
        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.vs_name, reuse=self.reuse) as vs:
            with slim.arg_scope([slim.fully_connected],
                                num_outputs=32,
                                activation_fn=__leaky_relu__):
                fc_1 = slim.fully_connected(self.input)
                fc_2 = slim.fully_connected(fc_1)
                fc_3 = slim.fully_connected(fc_2, num_outputs=2, activation_fn=None)

        self.output = fc_3
        self.var_list = tf.contrib.framework.get_variables(vs)


if __name__ == '__main__':
    pass
