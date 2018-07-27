import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb_hw, anchor_hw, iou  = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bbhw, _anchorhw, _iou = sess.run([bb_hw, anchor_hw, iou])
        print(_bbhw.shape)
        print(_bbhw)
        print(_anchorhw.shape)
        print(_anchorhw)
        print(_iou.shape)
        print(_iou)
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
