from __future__ import absolute_import


import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import Dataset
from Yolo import Yolo
import ConfigParser


class Train():
    def __init__(self):
        train_path = "./data/xview_train_t1.record"
        # val_path = "./data/eurosatTest.tfrecord"
        anchor_path= "./data/xview_anchors"

        self.config = ConfigParser.ConfigParser()
        self.config.read("config/conf.cfg")

        self.height = int(self.config.get("Common Params", "height"))
        self.width = int(self.config.get("Common Params", "width"))
        self.batch_size = int(self.config.get("Common Params", "batch_size"))
        self.num_epoch = int(self.config.get("Common Params", "num_epoch"))
        self.shuffle = int(self.config.get("Common Params", "shuffle"))
        self.train_dir = self.config.get("Common Params", "train_dir")
        self.max_iterations = int(self.config.get("Common Params", "max_iterations"))
        self.learning_rate =float(self.config.get("Common Params", "learning_rate"))

        self.trainDataset = Dataset(train_path, anchor_path)
        self.trainDataset.build(height=self.height, width=self.width, batch_size=self.batch_size, num_epochs=self.num_epoch,
                                shuffle=self.shuffle, num_parallel_calls=4)

        self.train_batch_number = len(self.trainDataset)/self.batch_size

        # self.valDataset = Dataset(val_path)
        # self.valDataset.build(height=self.height, width=self.width, batch_size=self.batch_size,
        #                         num_epochs=self.num_epoch,
        #                         shuffle=self.shuffle, num_parallel_calls=4)
        #
        # self.val_batch_number = len(self.valDataset)/self.batch_size


        self.yolo = Yolo2()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def solve(self):
        train_x, train_y, train_yohe = self.trainDataset.get_next()
        train_predict, train_softmax = self.yolo.inference(train_x)

        train_loss = self.yolo.loss(train_predict,train_yohe)

        training_step = self.optimizer.minimize(train_loss)

        train_loss_summ = tf.summary.scalar('loss', train_loss)

        train_acc, train_acc_op = tf.metrics.accuracy(labels=tf.argmax(train_yohe, 1),
                                                predictions=tf.argmax(train_predict, 1))

        train_acc_summ = tf.summary.scalar('train_accuracy', train_acc_op)

        train_summary = tf.summary.merge([train_loss_summ,train_acc_summ])

        val_x, val_y, val_yohe = self.valDataset.get_next()
        val_predict, val_softmax = self.yolo.inference(val_x, reuse = True)

        val_loss = self.yolo.loss(val_predict,val_yohe)


        val_acc, val_acc_op = tf.metrics.accuracy(labels=tf.argmax(val_yohe, 1),
                                                predictions=tf.argmax(val_predict, 1))

        val_loss_summ = tf.summary.scalar('loss', val_loss)
        val_acc_summ = tf.summary.scalar('val_accuracy', val_acc_op)

        val_summary = tf.summary.merge([val_loss_summ,val_acc_summ])

        saver = tf.train.Saver()

        print(len(self.trainDataset))
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        best_val_acc = 0
        with tf.Session() as sess:
            sess.run(init)
            sess.run(init_local)
            sess.run(self.trainDataset.init())
            sess.run(self.valDataset.init())
            trainWriter = tf.summary.FileWriter("./logs/train", sess.graph)
            valWriter = tf.summary.FileWriter("./logs/val")
            for epoch in range(self.num_epoch):
                print("\nEpoch:{}\n".format(epoch))
                train_progbar = tf.keras.utils.Progbar(self.train_batch_number)
                for step in xrange(self.train_batch_number):
                    _, _train_loss,_tr_acc,_tr_acc_op, _train_summary = sess.run([training_step,train_loss, train_acc, train_acc_op,train_summary])
                    train_progbar.update(step, [("tr_loss", _train_loss), ("tr_accuracy", _tr_acc_op)])
                print("\nValidation start\n")
                val_progbar = tf.keras.utils.Progbar(target=self.val_batch_number)
                for step in xrange(self.val_batch_number):
                    _val_loss,_val_acc,_val_acc_op,_val_summary = sess.run([val_loss,val_acc, val_acc_op, val_summary])
                    val_progbar.update(step, [("val_loss", _val_loss), ("val_accuracy", _val_acc_op)])
                trainWriter.add_summary(_train_summary,epoch)
                valWriter.add_summary(_val_summary)

                if _val_acc_op > best_val_acc:
                    best_val_acc = _val_acc_op
                    saver.save(sess,"./logs/yoloAcc_{}".format(_val_acc_op), global_step=epoch)
                print "\nBest val accuracy: {}\n".format(best_val_acc)


def main(argv=None):
    import shutil
    shutil.rmtree('./logs')
    train = Train()
    train.solve()

if __name__ == '__main__':
    tf.app.run()