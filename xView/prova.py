import tensorflow as tf
import matplotlib.pyplot as plt


from dataset import Dataset

dataset = Dataset("./data/xview_train_t1.record")

dataset.build()

img, bbox, label = dataset.get_next()

with tf.Session() as sess:
    sess.run(dataset.init())
    _img, _bbox, _label = sess.run([img, bbox, label])

    print(_label[0])
    print(_bbox[0])
    # plt.imshow(_img[0])
    # plt.show()
    print(_img[0].shape)