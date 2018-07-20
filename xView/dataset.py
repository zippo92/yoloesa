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
              shuffle=1000000, num_parallel_calls=2):
        self._num_class = num_class
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle = shuffle
        # self.dataset = self.dataset.shuffle(self._shuffle)
        self.dataset = self.dataset.map(self.__input_parser, num_parallel_calls=num_parallel_calls)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._batch_size))
        # self.dataset = self.dataset.repeat(self._num_epochs)
        self._iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                         self.dataset.output_shapes)

    def get_next(self):
        return self._iterator.get_next()

    def init(self):
        return self._iterator.make_initializer(self.dataset)

    def __input_parser(self, example):

        read_features = {
            'image/height': tf.FixedLenFeature((), dtype=tf.int64),
            'image/width': tf.FixedLenFeature((), dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature((), dtype=tf.string),
            'image/format': tf.FixedLenFeature((), dtype=tf.string),
            'image/object/bbox/xmin': tf.FixedLenFeature((), dtype=tf.float32),
            'image/object/bbox/xmax': tf.FixedLenFeature((), dtype=tf.float32),
            'image/object/bbox/ymin': tf.FixedLenFeature((), dtype=tf.float32),
            'image/object/bbox/ymax': tf.FixedLenFeature((), dtype=tf.float32),
            'image/object/class/label': tf.FixedLenFeature((), dtype=tf.int64)}

        parsed = tf.parse_single_example(example, features=read_features)

        img = tf.decode_raw(parsed['image/encoded'], tf.uint8)
        # img = tf.image.decode_jpeg(parsed['image/encoded'])
        width = parsed['image/width']
        height = parsed['image/height']
        format = parsed['image/format']
        xmin = parsed['image/object/bbox/xmin']
        xmax = parsed['image/object/bbox/xmax']
        ymin = parsed['image/object/bbox/ymin']
        ymax = parsed['image/object/bbox/ymax']
        label = parsed['image/object/class/label']

        bbox = [xmin,xmax,ymin,ymax]

        return img, bbox, label