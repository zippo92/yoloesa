from Dataset import Dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import ConfigParser

from Yolo import Yolo


class YoloSolver():

    def __init__(self):
        self.dataset = Dataset("eurosatDb.tfrecord")

        self.config = ConfigParser.ConfigParser()
        self.config.read("config/conf.cfg")

        self.height = int(self.config.get("Common Params", "height"))
        self.width = int(self.config.get("Common Params", "width"))
        self.batch_size = int(self.config.get("Common Params", "batch_size"))
        self.num_epoch = int(self.config.get("Common Params", "num_epoch"))
        self.shuffle = int(self.config.get("Common Params", "shuffle"))
        self.train_dir = self.config.get("Common Params", "train_dir")
        self.max_iterations = int(self.config.get("Common Params", "max_iterations"))

        self.dataset.build(height=self.height, width=self.width, batch_size=self.batch_size, num_epoch=self.num_epoch, shuffle=self.shuffle)

        self.yolo = Yolo()
        self.construct_graph()



    def construct_graph(self):
        self.images, self.labels, self.labelsohe = self.dataset.get_next()
        self.predicts, self.logits = self.yolo.inference(self.images)
        self.total_loss = self.yolo.loss(self.logits, self.labelsohe)
        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self._train(self.total_loss)


    def _train(self, loss):
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
        config.read("config/conf.cfg")

        learning_rate =float(config.get("Common Params", "learning_rate"))
        moment = float(config.get("Common Params", "moment"))
        opt = tf.train.AdamOptimizer()
        train_step = opt.minimize(loss)
        return train_step

            # grads = opt.compute_gradients(self.total_loss)

            # apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

            #return apply_gradient_op


    def solve(self):
        init = tf.global_variables_initializer()

        initDataset = self.dataset.init()

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:

            sess.run(init)
            sess.run(initDataset)

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            for epoch in xrange(self.num_epoch):
                # start_time = time.time()
                print("epoch:{}".format(epoch))

                progbar = tf.keras.utils.Progbar(len(self.dataset)/self.batch_size)

                for step in range(len(self.dataset)/self.batch_size):

                    _, loss_value,_summaryop = sess.run([self.train_op, self.total_loss, summary_op])
                    progbar.update(step,[("loss",loss_value)])
                    summary_writer.add_summary(_summaryop,epoch)

def main(argv=None):
    yolosolver = YoloSolver()
    yolosolver.solve()

if __name__ == '__main__':
    tf.app.run()
