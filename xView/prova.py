import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb, anchors = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bb, _anchor = sess.run([bb,anchors])
        print(_bb)

        print(_anchor)
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
