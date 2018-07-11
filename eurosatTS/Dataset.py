import os
import rasterio
import numpy as np
import tensorflow as tf

class Dataset():


    def __init__(self, root_dir):
        self._tfRecordpPath = root_dir
        self.dataset = tf.data.TFRecordDataset(self._tfRecordpPath)

    def build(self, height = 128, width = 128, batch_size = 12, num_epoch = 1, shuffle = 1800*12, num_parallel_calls = 4):

        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._num_epoch = num_epoch
        self._shuffle = shuffle

        self.dataset = self.dataset.shuffle(self._shuffle)
        self.dataset = self.dataset.map(self.parseTFRecord, num_parallel_calls)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self._batch_size))
        self.dataset = self.dataset.repeat(self._num_epoch)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)

    def init(self):
        return self.iterator.make_initializer(self.dataset)


    def get_next(self):
        return self.iterator.get_next()

    def __len__(self):
        return sum(1 for _ in tf.python_io.tf_record_iterator(self._tfRecordpPath))


    def parseTFRecord(self, example):
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
