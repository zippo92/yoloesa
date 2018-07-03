import os
import rasterio
import numpy as np
import tensorflow as tf

class Dataset():

    path = None
    dataset = None

    def __init__(self, root_dir):
        self.path = root_dir

    def createDataset(self):
        self.dataset = tf.data.TFRecordDataset(self.path)
        self.dataset = self.dataset.map(self.parseTFRecord)
        #self.dataset = self.dataset.batch(1)

        # self.dataset = self.dataset.map(self.parse_function, num_parallel_calls=4)
        # self.dataset = self.dataset.shuffle(len(self.x))
        # self.dataset = self.dataset.map(self.parse_function, num_parallel_calls=4)
        #self.dataset = self.dataset.batch(2)
        # self.dataset = self.dataset.prefetch(1)
        #

        # # step 4: create iterator and final input tensor
        iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        images, labels = iterator.get_next()
        init = iterator.make_initializer(self.dataset)
        with tf.Session() as sess:
            sess.run(init)
            _img, _label = sess.run([images, labels])
            print _img.shape

    def __len__(self):
        return sum(1 for _ in tf.python_io.tf_record_iterator(self.path))


    def parseTFRecord(self, example):
        read_features = {
            'image': tf.FixedLenFeature((), dtype=tf.string),
            'label': tf.FixedLenFeature((), dtype=tf.int64)}

        parsed = tf.parse_single_example(example, features=read_features)

        img = tf.decode_raw(parsed['image'], tf.float16)
        label = parsed['label']

        img = tf.reshape(img, [64, 64,3])


        return img,label

