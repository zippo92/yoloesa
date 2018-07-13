from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os

class Dataset(object):
    def __init__(self, file_path):
        self.dataset = tf.data.TFRecordDataset(file_path)

    def __len__(self):
        return sum(1 for _ in tf.python_io.tf_record_iterator(self._tfRecordpPath))

    def build(self, num_class=10,
              height=256, width=256,
              batch_size=10, num_epochs=1,
              shuffle=1000000, num_parallel_calls=2):
        self._num_class = num_class
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle = shuffle
        self.dataset = self.dataset.shuffle(self._shuffle)
        self.dataset = self.dataset.map(self.__input_parser, num_parallel_calls=num_parallel_calls)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._batch_size))
        self.dataset = self.dataset.repeat(self._num_epochs)
        self._iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                         self.dataset.output_shapes)

    def get_next(self):
        return self._iterator.get_next()

    def init(self):
        return self._iterator.make_initializer(self.dataset)

    def __input_parser(self, example):

        read_features = {
            'image': tf.FixedLenFeature((), dtype=tf.string),
            'label': tf.FixedLenFeature((), dtype=tf.int64)}

        parsed = tf.parse_single_example(example, features=read_features)

        img = tf.decode_raw(parsed['image'], tf.float32)
        label = parsed['label']

        img = tf.reshape(img, [64, 64,3])

        img = tf.image.resize_images(img, [320, 320])

        label_ohe = tf.one_hot(label, 10)

        return img,label, label_ohe
