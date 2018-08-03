import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

img, det_mask, true_boxes = dataset.get_next()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    sess.run(init)
    sess.run(init_local)
    # sess.run(dataset.init())
    for i in range(10):
        _img, _det_mask, _true_boxes = sess.run([img, det_mask, true_boxes])
        print(_det_mask.shape)
        print(_true_boxes.shape)

        # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
