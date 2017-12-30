import os

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import data_generator
from models import NN_MAP

flags = tf.app.flags
flags.DEFINE_integer("n_batch_size", 512, "Batch size to train [512]")
FLAGS = flags.FLAGS


class Trainer(object):
    def __init__(self):
        self.x_generator = None
        self.y_generator = None
        self.x = None
        self.f = None
        self.fx = None
        self.f_var_list = None
        self.loss = None
        self.ckpt_dir_ot = 'ckpts/stochastic_ot_computation/'
        self.ckpt_dir_map = 'ckpts/optimal_map_estimation/'
        self.visualize_dir_map = 'viz/'
        self.f_saver = None
        self.sess = None
        self.coord = None
        self.threads = None

        self.define_dataset()
        self.define_model()
        self.define_saver()
        self.define_viz_dir()
        self.initialize_session_and_etc()

    def define_dataset(self):
        self.x_generator = iter(data_generator.GeneratorGaussian1(FLAGS.n_batch_size))
        self.y_generator = iter(data_generator.GeneratorGaussians4(FLAGS.n_batch_size))
        self.x = tf.placeholder(tf.float32, (None, 2))

    def define_model(self):
        self.f = NN_MAP(self.x, 'f')
        self.fx = self.f.output
        self.f_var_list = self.f.var_list

    def define_saver(self):
        self.f_saver = tf.train.Saver(self.f_var_list)

    def define_viz_dir(self):
        if not os.path.exists(self.visualize_dir_map):
            os.makedirs(self.visualize_dir_map)

    def initialize_session_and_etc(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.f_saver.restore(self.sess, self.ckpt_dir_map)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def train(self):
        try:
            x = next(self.x_generator)
            y = next(self.y_generator)

            fx = self.sess.run(self.fx, feed_dict={self.x: x})

            visualize(x, y, fx)

        except KeyboardInterrupt:
            print("Interrupted!")
            self.coord.request_stop()

        finally:
            self.f_saver.save(self.sess, self.ckpt_dir_map)
            print('Stop')
            self.coord.request_stop()
            self.coord.join(self.threads)


def visualize(x, y, fx):
    plt.scatter(x[:, 0], x[:, 1], s=1, c='g')
    plt.scatter(y[:, 0], y[:, 1], s=1, c='r')
    plt.xlim(-1.5, +1.5)
    plt.ylim(-1.5, +1.5)
    plt.savefig('viz/XnY.png')
    plt.clf()

    plt.scatter(x[:, 0], x[:, 1], s=1, c='g')
    plt.scatter(fx[:, 0], fx[:, 1], s=1, c='b')

    ax = plt.axes()
    for i in range(int(x.shape[0]/8)):
        ax.arrow(x[i, 0], x[i, 1], fx[i, 0]-x[i, 0], fx[i, 1]-x[i, 1],
                 head_width=0.03, head_length=0.02, fc='k', ec='k')

    plt.xlim(-1.5, +1.5)
    plt.ylim(-1.5, +1.5)

    plt.savefig('viz/XnFx.png')
    plt.clf()

    fig = sns.jointplot(fx[:, 0], fx[:, 1], kind='kde')
    fig.savefig('viz/Fx.png')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("Done!")
