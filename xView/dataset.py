from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import ast


class Dataset(object):
    def __init__(self, file_path, anchor_path):
        self.path = file_path
        self.anchor_path = anchor_path
        self.dataset = tf.data.TFRecordDataset(file_path)

    def __len__(self):
        return sum(1 for _ in tf.python_io.tf_record_iterator(self.path))

    def build(self, num_class=10,
              height=416, width=416,
              batch_size=32, num_epochs=100,
              shuffle=1000000, num_parallel_calls=2, s=9, b=4):
        self._num_class = num_class
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle = shuffle
        self._s = s
        self._b = b
        with open(self.anchor_path, 'r') as f:
            self._anchors = ast.literal_eval(f.read())
        self.dataset = self.dataset.shuffle(self._shuffle)
        self.dataset = self.dataset.map(self.__input_parser, num_parallel_calls=num_parallel_calls)

        # self.dataset = self.dataset.padded_batch(self._batch_size, padded_shapes= [None,None,None])
        # self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._batch_size))
        # self.dataset = self.dataset.repeat(self._num_epochs)
        # self._iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
        #                                                        self.dataset.output_shapes)

        self._iterator = self.dataset.make_initializable_iterator()

    def get_next(self):
        return self._iterator.get_next()

    def init(self):
        # return self._iterator.make_initializer(self.dataset)
        return self._iterator.initializer

    def __parse_bb(self,tensor):
        xmin = tensor[0]
        xmax = tensor[1]
        ymin = tensor[2]
        ymax = tensor[3]
        box_class = tensor[4]

        width = tf.subtract(xmax, xmin)
        height = tf.subtract(ymax, ymin)

        x_center = tf.add(xmin, tf.divide(width,tf.constant(2,dtype=tf.float32)))
        y_center = tf.add(ymin, tf.divide(height,tf.constant(2, dtype=tf.float32)))




        return x_center, y_center, height, width, box_class

    def __input_parser(self, example):
        read_features = {
            'image/height': tf.FixedLenFeature((), dtype=tf.int64),
            'image/width': tf.FixedLenFeature((), dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature((), dtype=tf.string),
            'image/format': tf.FixedLenFeature((), dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64)}

        parsed = tf.parse_single_example(example, features=read_features)

        img = tf.image.decode_image(parsed['image/encoded'])
        width = parsed['image/width']
        height = parsed['image/height']
        format = parsed['image/format']
        bb_xmin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmin'])
        bb_xmax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmax'])
        bb_ymin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymin'])
        bb_ymax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymax'])
        label = tf.sparse_tensor_to_dense(parsed['image/object/class/label'])
        label = tf.cast(label, tf.float32)

        bb = tf.stack([bb_xmin,bb_xmax,bb_ymin,bb_ymax, label], axis = 1)

        conv_height = self._height // 32
        conv_width = self._width // 32
        num_box_params = bb.get_shape()[1]
        num_anchors = len(self._anchors)
        detector_mask =tf.zeros([conv_height,conv_width, num_anchors, 1], dtype=tf.float32)
        matching_true_boxes = tf.zeros([conv_height,conv_width,num_anchors, num_box_params])
        img = tf.reshape(img,[self._height,self._width,3])
        img = tf.image.resize_images(img, size = [self._height,self._width])


        bb = tf.map_fn(self.__parse_bb, bb, dtype = (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))

        grid_x = tf.floor(tf.multiply(bb[0], tf.constant(self._s, dtype=tf.float32)))
        grid_y = tf.floor(tf.multiply(bb[1], tf.constant(self._s, dtype=tf.float32)))

        bb_hw = tf.stack([bb[2], bb[3]], axis = 1) #num_bb, h, w
        anchors = tf.convert_to_tensor(self._anchors, dtype=tf.float32) #num_anchor, h, w

        anchors_hw = tf.tile(tf.expand_dims(anchors , axis=0),[tf.shape(bb_hw)[0], 1, 1])
        bb_hw = tf.tile(tf.expand_dims(bb_hw, axis = 0), [tf.shape(anchors)[0], 1, 1])

        print(bb_hw)
        print(anchors_hw)



        # grid_x_offset = tf.subtract(grid_x, tf.round(grid_x))
        # grid_y_offset = tf.subtract(grid_y, tf.round(grid_y))

        # Per ogni active anchor
        #     anchor_label=[grid_x_offset, grid_y_offset, bb[4] , bb[5]]
        #     zeros[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [label], [1.0]) #TODO equivalente con tf.scatter_update



        return bb_hw, anchors_hw
