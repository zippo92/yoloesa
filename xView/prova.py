import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb_stack, grid_x_offset  = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bb_stack, _grid_x_offset = sess.run([bb_stack, grid_x_offset])
        print(_bb_stack.shape)
        print(_bb_stack)
        print(_grid_x_offset.shape)
        print(_grid_x_offset[8])
      
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
