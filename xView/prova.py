import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record", "./data/xview_anchors")

dataset.build()

bb_stack, grid_x_offset, grid_y_offset, log1, log2  = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    for i in range(1):
        _bb_stack, _grid_x_offset, _grid_y_offset, _log1, _log2 = sess.run([bb_stack, grid_x_offset, grid_y_offset, log1, log2])
        print(_bb_stack.shape)
        print(_bb_stack)
        print(_grid_x_offset.shape)
        print(_grid_x_offset)
        print(_grid_y_offset.shape)
        print(_grid_y_offset)
        print(_log1.shape)
        print(_log1)
        print(_log2.shape)
        print(_log2
              )

	
    # plt.imshow(_img[0])
    # plt.show()
# print(_img[0].shape)
