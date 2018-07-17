from __future__ import absolute_import


import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import Dataset
from Yolo import Yolo

class Train():
    def __init__(self):
        path = "./data/eurosatDb.tfrecord"

        self.dataset = Dataset(path)
        self.dataset.build()
        self.yolo = Yolo()

    def solve(self):

        loss, training_step = self.construct_graph()

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

    def construct_graph(self):
        images, labels, labelsohe = self.dataset.get_next()
        predicts, logits = self.yolo.inference(images)
        total_loss = self.yolo.loss(logits, labelsohe)

        optimizer = tf.train.AdamOptimizer()
        training_step = optimizer.minimize(self.loss)

        return total_loss, training_step
def main(argv=None):
    train = Train()
    train.solve()

if __name__ == '__main__':
    tf.app.run()