import os

import tensorflow as tf

import data_generator
from models import NN_DUAL, NN_MAP
from losses import primer_dual

flags = tf.app.flags
flags.DEFINE_integer("n_epoch", 20000, "Epoch to train [20000]")
flags.DEFINE_integer("n_batch_size", 512, "Batch size to train [512]")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of optimizer [0.005]")
FLAGS = flags.FLAGS


class Trainer(object):
    def __init__(self):
        self.x_generator = None
        self.y_generator = None
        self.x = None
        self.y = None
        self.u = None
        self.v = None
        self.ux = None
        self.vy = None
        self.f = None
        self.fx = None
        self.u_var_list = None
        self.v_var_list = None
        self.f_var_list = None
        self.loss = None
        self.step = None
        self.step_inc = None
        self.f_opt = None
        self.ckpt_dir_ot = 'ckpts/stochastic_ot_computation/'
        self.ckpt_dir_map = 'ckpts/optimal_map_estimation/'
        self.summary_writer = None
        self.summary_op = None
        self.u_saver = None
        self.v_saver = None
        self.f_saver = None
        self.sess = None
        self.coord = None
        self.threads = None

        self.define_dataset()
        self.define_model()
        self.define_loss()
        self.define_optim()
        self.define_writer_and_summary()
        self.define_saver()
        self.initialize_session_and_etc()

    def define_dataset(self):
        self.x_generator = iter(data_generator.GeneratorGaussian1(FLAGS.n_batch_size))
        self.y_generator = iter(data_generator.GeneratorGaussians4(FLAGS.n_batch_size))
        self.x = tf.placeholder(tf.float32, (None, 2))
        self.y = tf.placeholder(tf.float32, (None, 2))

    def define_model(self):
        self.u = NN_DUAL(self.x, 'u')
        self.v = NN_DUAL(self.y, 'v')
        self.ux = self.u.output
        self.vy = self.v.output
        self.u_var_list = self.u.var_list
        self.v_var_list = self.v.var_list

        self.f = NN_MAP(self.x, 'f')
        self.fx = self.f.output
        self.f_var_list = self.f.var_list

    def define_loss(self):
        self.loss = tf.reduce_mean(primer_dual(self.fx, self.ux, self.vy, self.x, self.y))

    def define_optim(self):
        self.step = tf.Variable(0, name='step', trainable=False)
        self.step_inc = tf.assign(self.step, self.step + 1)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.f_opt = optimizer.minimize(self.loss, var_list=self.f_var_list)

    def define_writer_and_summary(self):
        if not os.path.exists(self.ckpt_dir_ot):
            os.makedirs(self.ckpt_dir_ot)

        if not os.path.exists(self.ckpt_dir_map):
            os.makedirs(self.ckpt_dir_map)

        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir_map)

        with tf.control_dependencies([self.f_opt]):
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('loss', self.loss)
            ])

    def define_saver(self):
        self.u_saver = tf.train.Saver(self.u_var_list)
        self.v_saver = tf.train.Saver(self.v_var_list)
        self.f_saver = tf.train.Saver(self.f_var_list)

    def initialize_session_and_etc(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.u_saver.restore(self.sess, self.ckpt_dir_ot)
        self.v_saver.restore(self.sess, self.ckpt_dir_ot)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def train(self):
        try:
            step = None
            while not self.coord.should_stop():
                step = self.sess.run(self.step)
                if step > FLAGS.n_epoch:
                    break

                x = next(self.x_generator)
                y = next(self.y_generator)

                summary = self.sess.run(self.summary_op, feed_dict={self.x: x, self.y: y})
                self.summary_writer.add_summary(summary, step)
                self.summary_writer.flush()

                self.sess.run(self.step_inc)

        except KeyboardInterrupt:
            print("Interrupted!")
            self.coord.request_stop()

        finally:
            self.f_saver.save(self.sess, self.ckpt_dir_map)
            print('Stop')
            self.coord.request_stop()
            self.coord.join(self.threads)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print("Done!")
