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

        img = tf.image.decode_jpeg(parsed['image/encoded'])
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
        #img = img.set_shape([height,width,3])
        img = tf.image.resize_images(img, size = [self._height,self._width])
	img = tf.image.convert_image_dtype(img,dtype = tf.float32)

        bb = tf.map_fn(self.__parse_bb, bb, dtype = (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))

        grid_x = tf.floor(tf.multiply(bb[0], tf.constant(conv_width, dtype=tf.float32)))
        grid_y = tf.floor(tf.multiply(bb[1], tf.constant(conv_height, dtype=tf.float32)))

        bb_hw = tf.stack([bb[2], bb[3]], axis = 1) #num_bb, h, w
        anchors = tf.convert_to_tensor(self._anchors, dtype=tf.float32) #num_anchor, h, w
        anchors_hw = tf.tile(tf.expand_dims(anchors , axis=0),[tf.shape(bb_hw)[0], 1, 1])
        bb_hw = tf.tile(tf.expand_dims(bb_hw, axis = 1), [1, tf.shape(anchors)[0], 1])


        box_maxes = bb_hw / 2.
        box_mins = -box_maxes
        anchor_maxes = (anchors_hw / 2.)
        anchor_mins = -anchor_maxes

        intersect_mins = tf.maximum(box_mins, anchor_mins)
        intersect_maxes = tf.minimum(box_maxes, anchor_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[:,:,0] * intersect_wh[:,:,1]
        box_area = bb_hw[:,:,0]*bb_hw[:,:,1]
        anchor_area = anchors_hw[:,:,0] * anchors_hw[:,:,1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        iou_max = tf.reduce_max(iou, axis=[1])
        iou_argmax = tf.argmax(iou, dimension=1)

        condition = tf.less(tf.constant(0.5), iou_max)
        non_zeros = tf.where(condition)
	non_zeros = tf.squeeze(non_zeros)
        iou_argmax = tf.cast(iou_argmax, tf.float32)

        iou_stack = tf.stack([tf.cast(grid_x, tf.int32), tf.cast(grid_y,tf.int32), tf.cast(iou_argmax, tf.int32)], axis=1)
        iou_stack = tf.gather(iou_stack,non_zeros, axis = 0)

        updates = tf.ones(shape=(tf.shape(iou_stack)[0]))

        shape = tf.constant([conv_height, conv_width,num_anchors])
	mask = tf.scatter_nd(iou_stack,updates, shape)
	return bb_hw,  anchors_hw, iou, iou_stack, mask
