import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb_hw, anchor_hw, iou, iou_max, iou_argmax  = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bbhw, _anchorhw, _iou, _iou_max, _iou_argmax = sess.run([bb_hw, anchor_hw, iou, iou_max, iou_argmax])
        print(_bbhw.shape)
        print(_bbhw)
        print(_anchorhw.shape)
        print(_anchorhw)
        print(_iou.shape)
        print(_iou)
	print(_iou_max.shape)
	print(_iou_max)
	print(_iou_argmax.shape)
	print(_iou_argmax[10])

	
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
