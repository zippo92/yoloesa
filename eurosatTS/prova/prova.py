from __future__ import absolute_import


import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dataset import Dataset

path = "./data/eurosatDb.tfrecord"

dataset = Dataset(path)
dataset.build()
x,y,yohe = dataset.get_next()

conv1 = tf.layers.conv2d(inputs=x,filters=64,kernel_size=3,padding="same",strides = 1)
activation1 = tf.nn.relu(conv1)

conv2 = tf.layers.conv2d(inputs=activation1,filters=64,kernel_size=3,padding="same",strides = 1)
activation2 = tf.nn.relu(conv2)

max1 = tf.layers.max_pooling2d(inputs=activation2, pool_size=2, strides=2)

conv3 = tf.layers.conv2d(inputs=max1,filters=128,kernel_size=3,padding="same",strides = 1)
activation3 = tf.nn.relu(conv3)

conv4 = tf.layers.conv2d(inputs=activation3,filters=128,kernel_size=3,padding="same",strides = 1)
activation4 = tf.nn.relu(conv4)

max2 = tf.layers.max_pooling2d(inputs=activation4, pool_size=2, strides=2)

conv5 = tf.layers.conv2d(inputs=max2,filters=256,kernel_size=3,padding="same",strides = 1)
activation5 = tf.nn.relu(conv5)

conv6 = tf.layers.conv2d(inputs=activation5,filters=256,kernel_size=3,padding="same",strides = 1)
activation6 = tf.nn.relu(conv6)

conv7 = tf.layers.conv2d(inputs=activation6,filters=256,kernel_size=3,padding="same",strides = 1)
activation7 = tf.nn.relu(conv7)

max3 = tf.layers.max_pooling2d(inputs=activation7, pool_size=2, strides=2)

conv8 = tf.layers.conv2d(inputs=max3,filters=512,kernel_size=3,padding="same",strides = 1)
activation8 = tf.nn.relu(conv8)

conv9 = tf.layers.conv2d(inputs=activation8,filters=512,kernel_size=3,padding="same",strides = 1)
activation9 = tf.nn.relu(conv9)

conv10 = tf.layers.conv2d(inputs=activation9,filters=512,kernel_size=3,padding="same",strides = 1)
activation10 = tf.nn.relu(conv10)

max4 = tf.layers.max_pooling2d(inputs=activation10, pool_size=2, strides=2)

conv11 = tf.layers.conv2d(inputs=max4,filters=512,kernel_size=3,padding="same",strides = 1)
activation11 = tf.nn.relu(conv11)

conv12 = tf.layers.conv2d(inputs=activation11,filters=512,kernel_size=3,padding="same",strides = 1)
activation12 = tf.nn.relu(conv12)

conv13 = tf.layers.conv2d(inputs=activation12,filters=512,kernel_size=3,padding="same",strides = 1)
activation13 = tf.nn.relu(conv13)

max5 = tf.layers.max_pooling2d(inputs=activation13, pool_size=2, strides=2)

flat = tf.layers.flatten(max5)

dense1 = tf.layers.dense(inputs=flat, units=4096)

activation14 = tf.nn.relu(dense1)

dense2 = tf.layers.dense(inputs=activation14, units=10)

softmax = tf.nn.softmax(dense2)

loss = tf.losses.softmax_cross_entropy(onehot_labels=yohe, logits=dense2)

optimizer = tf.train.AdamOptimizer()
training_step = optimizer.minimize(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(dataset.init())
    for i in range(30):
        _, _loss = sess.run([training_step, loss])
        print(i, _loss)
