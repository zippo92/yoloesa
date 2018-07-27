import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb_hw1, bb_hw2 = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bbhw1, _bbhw2 = sess.run([bb_hw1, bb_hw2])
        print(bb_hw1.shape)
        print(bb_hw1)
        print(bb_hw2.shape)
        print(bb_hw2)
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
