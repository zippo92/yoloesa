from __future__ import absolute_import


import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import Dataset
from Yolo import Yolo
import ConfigParser


class Train():
    def __init__(self):
        path = "./data/eurosatDb.tfrecord"

        self.config = ConfigParser.ConfigParser()
        self.config.read("config/conf.cfg")

        self.height = int(self.config.get("Common Params", "height"))
        self.width = int(self.config.get("Common Params", "width"))
        self.batch_size = int(self.config.get("Common Params", "batch_size"))
        self.num_epoch = int(self.config.get("Common Params", "num_epoch"))
        self.shuffle = int(self.config.get("Common Params", "shuffle"))
        self.train_dir = self.config.get("Common Params", "train_dir")
        self.max_iterations = int(self.config.get("Common Params", "max_iterations"))

        self.dataset = Dataset(path)

        self.dataset.build(height=self.height, width=self.width, batch_size=self.batch_size, num_epochs=self.num_epoch,
                           shuffle=self.shuffle, num_parallel_calls=4)

        # self.dataset.build()
        self.yolo = Yolo()

    def solve(self):
        x, y, yohe = self.dataset.get_next()
        dense2, softmax = self.yolo.inference(x)

        loss = self.yolo.loss(dense2,yohe)
        optimizer = tf.train.AdamOptimizer()
        training_step = optimizer.minimize(loss)

        print(len(self.dataset))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(self.dataset.init())
            for epoch in range(100):
                print("Epoch:{}".format(epoch))
                progbar = tf.keras.utils.Progbar(675)
                for step in range(675):
                    _, _loss = sess.run([training_step, loss])
                    progbar.update(step, [("loss", _loss)])


def main(argv=None):
    train = Train()
    train.solve()

if __name__ == '__main__':
    tf.app.run()