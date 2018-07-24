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

        width = tf.subtract(xmax, xmin)
        height = tf.subtract(ymax, ymin)

        x_center = tf.add(xmin, tf.divide(width,tf.constant(2,dtype=tf.float32)))
        y_center = tf.add(ymin, tf.divide(height,tf.constant(2, dtype=tf.float32)))

        # Bisogna dividere width e height per gli anchor attivi, gli anchor dovranno essere sicuramente in percentuale

        # anchor_indxs = self.get_active_anchors(width, height)
        # w_scale = tf.divide(width,anchor[0])
        # h_scale = tf.divide(height,anchor[1])

        return width, height, x_center,y_center #w_scale,h_scale

    #TODO da fare, l'ho solo copiata
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

        grid_x = tf.multiply(bb[2], tf.constant(self._s, dtype=tf.float32))
        grid_y = tf.multiply(bb[3], tf.constant(self._s, dtype=tf.float32))

        grid_x_offset = tf.subtract(grid_x, tf.round(grid_x))
        grid_y_offset = tf.subtract(grid_y, tf.round(grid_y))


        # Per ogni active anchor
        #     anchor_label=[grid_x_offset, grid_y_offset, bb[4] , bb[5]]
        #     zeros[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [label], [1.0]) #TODO equivalente con tf.scatter_update



        return img, grid_x, grid_y, bb[2], bb[3], label
