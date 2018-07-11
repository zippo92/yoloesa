from Dataset import Dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import ConfigParser

from .yoloNet.Yolo import Yolo

class YoloSolver():

    def __init__(self):
        self.dataset = Dataset("eurosatDb.tfrecord")

        self.config = ConfigParser.ConfigParser()
        self.config.read("../config/conf.cfg")

        self.height = self.config.get("Common Params", "height")
        self.width = self.config.get("Common Params", "width")
        self.batch_size = self.config.get("Common Params", "batch_size")
        self.num_epoch = self.config.get("Common Params", "num_epoch")
        self.shuffle = self.config.get("Common Params", "shuffle")
        self.train_dir = self.config.get("Common Params", "train_dir")
        self.max_iterations = self.config.get("Common Params", "max_iterations")

        self.dataset.build(height=self.height, width=self.width, batch_size=self.batch_size, num_epoch=self.num_epoch, shuffle=self.shuffle)

        print(len(self.dataset) / 12)

        self.construct_graph()
        self.yolo = Yolo()



    def construct_graph(self):

        # construct graph
        self.global_step = tf.Variable(0, trainable=False)
        # self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
        # self.labels = tf.placeholder(tf.float32, (self.batch_size, 10)) # TODO verificare

        self.images, self.labels, self.labelsohe = self.dataset.get_next()

        self.predicts, self.logits = self.yolo.inference(self.images, mode=tf.estimator.ModeKeys.TRAIN)
        self.total_loss = self.yolo.loss(self.logits, self.labelsohe)

        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self._train()


    def _train(self):
            """Train model
            Create an optimizer and apply to all trainable variables.
            Args:
              total_loss: Total loss from net.loss()
              global_step: Integer Variable counting the number of training steps
              processed
            Returns:
              train_op: op for training
            """
            config = ConfigParser.ConfigParser()
            config.read("../config/conf.cfg")

            learning_rate = config.get("Common Params", "learning_rate")
            moment = config.get("Common Params", "moment")
            opt = tf.train.MomentumOptimizer(learning_rate, moment)
            grads = opt.compute_gradients(self.total_loss)

            apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

            return apply_gradient_op


    def solve(self):
        init = tf.global_variables_initializer()

        self.initDataset = self.dataset.init()

        summary_op = tf.summary.merge_all()

        sess = tf.Session()

        sess.run(init)

        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

        for step in xrange(self.max_iterators):
            # start_time = time.time()
            np_images, np_labels = self.dataset.get_next()

            _, loss_value,_summaryop = sess.run([self.train_op, self.total_loss, summary_op])

            print(step, loss_value)
            summary_writer.add_summary(_summaryop,step)

def main(argv=None):
    yolosolver = YoloSolver()
    yolosolver.solve()

if __name__ == '__main__':
    tf.app.run()
