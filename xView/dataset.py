from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os


class Dataset(object):
    def __init__(self, file_path):
        self.path = file_path
        self.dataset = tf.data.TFRecordDataset(file_path)

    def __len__(self):
        return sum(1 for _ in tf.python_io.tf_record_iterator(self.path))

    def build(self, num_class=10,
              height=256, width=256,
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
        # self.dataset = self.dataset.shuffle(self._shuffle)
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

        width = tf.subtract(xmax, xmin)
        height = tf.subtract(ymax, ymin)

        # acnhor_indxs = self.get_active_anchors(width, height)
        x_center = tf.divide(tf.add(xmin,width),tf.constant(2))
        y_center = tf.divide(tf.add(ymin,height),tf.constant(2))

        return width, height, x_center,y_center
    def get_active_anchors(self, w, h):
        indxs = []
        iou_max, index_max = 0, 0
        for i, a in enumerate(self.anchors):
            iou = self.iou_wh([w,h], a)
            if iou > 0.7:
                indxs.append(i)
            if iou > iou_max:
                iou_max, index_max = iou, i

        if len(indxs) == 0:
            indxs.append(index_max)

        return indxs

    def iou_wh(self, r1, r2):
        min_w = tf.minimum(r1[0], r2[0])
        min_h = tf.minimum(r1[1], r2[1])
        area_r1 = r1[0] * r1[1]
        area_r2 = r2[0] * r2[1]

        intersect = min_w * min_h
        union = area_r1 + area_r2 - intersect

        return intersect / union

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

        zeros = tf.zeros(shape=[self._s,self._s,self._b*4+self._num_class])
        parsed = tf.parse_single_example(example, features=read_features)

        img = tf.image.decode_image(parsed['image/encoded'])
        # img = tf.image.decode_jpeg(parsed['image/encoded'])
        width = parsed['image/width']
        height = parsed['image/height']
        format = parsed['image/format']
        bb_xmin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmin'])
        bb_xmax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmax'])
        bb_ymin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymin'])
        bb_ymax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymax'])
        label = tf.sparse_tensor_to_dense(parsed['image/object/class/label'])

        bb = tf.stack([bb_xmin,bb_xmax,bb_ymin,bb_ymax], axis = 1)

        bb = tf.map_fn(self.__parse_bb, bb, dtype = (tf.float32,tf.float32,tf.float32,tf.float32))

        raw_w = tf.shape(img)[0]
        raw_h = tf.shape(img)[1]
        grid_x = tf.multiply(tf.divide(bb[3],raw_w), width)
        grid_y = tf.multiply(tf.divide(bb[4],raw_h), height)

        return img, grid_x, label
