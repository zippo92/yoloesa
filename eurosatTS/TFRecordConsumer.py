import os
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # Read and print data:
    sess = tf.InteractiveSession()

    # Read TFRecord file
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(['eurosatDb.tfrecord'])

    _, serialized_example = reader.read(filename_queue)

    # Define features
    read_features = {
        'Image': tf.FixedLenFeature([], dtype=tf.string),
        'Label': tf.FixedLenFeature([], dtype=tf.int64)}

    parsed = tf.parse_single_example(serialized_example,read_features)

    img = tf.decode_raw(parsed['Image'], tf.uint8)
    label = parsed['Label']
    # label = tf.decode_raw(parsed['Label'], tf.int64)

    print(img,label)