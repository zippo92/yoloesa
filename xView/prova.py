import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb, anchors, iou, iou_max, iou_stack = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bb, _anchor, _iou, _iou_max, _iou_stack = sess.run([bb,anchors, iou, iou_max, iou_stack])
        print(_bb.shape)
        print(_anchor.shape)
        print(_bb)

        print(_anchor)

        print(_iou)

        print(_iou_max)

        print(_iou_stack)
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
