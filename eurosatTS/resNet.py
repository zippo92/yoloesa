import tensorflow as tf


class resNet():

    def __init__(self):
        pass

    def inference(self, images, dropout = True, reuse=False):

        with tf.variable_scope('yolo', reuse=reuse):
            predicts = self.conv2d(images, filters=64, kernel_size=(7, 7), stride=(2, 2))
            predicts = self.batch_normalization(predicts)
            predicts = tf.nn.relu(predicts)
            predicts = self.max_pool(predicts, kernel_size=(3, 3), strides=(2, 2))

            predicts = self.conv_block(predicts, filters=[64, 64, 256], strides=(1, 1))
            predicts = self.id_block(predicts, filters=[64, 64, 256])
            predicts = self.id_block(predicts, filters=[64, 64, 256])

            predicts = self.conv_block(predicts, filters=[128, 128, 512])
            predicts = self.id_block(predicts, filters=[128, 128, 512])
            predicts = self.id_block(predicts, filters=[128, 128, 512])
            predicts = self.id_block(predicts, filters=[128, 128, 512])

            predicts = self.conv_block(predicts, filters=[256, 256, 1024])
            predicts = self.id_block(predicts, filters=[256, 256, 1024])
            predicts = self.id_block(predicts, filters=[256, 256, 1024])
            predicts = self.id_block(predicts, filters=[256, 256, 1024])
            predicts = self.id_block(predicts, filters=[256, 256, 1024])
            predicts = self.id_block(predicts, filters=[256, 256, 1024])

            predicts = self.conv_block(predicts, filters=[512, 512, 2048])
            predicts = self.id_block(predicts, filters=[512, 512, 2048])
            predicts = self.id_block(predicts, filters=[512, 512, 2048])

            predicts = tf.layers.average_pooling2d(predicts, pool_size=(2, 2), strides=(2, 2))

            predicts = self.dense(predicts, units=1000, activation="relu", training=reuse, dropout=dropout)
            predicts = self.dense(predicts, units=10, activation="None", training=reuse, dropout=dropout)

            return predicts

    def conv_block(self, input, filters, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        x = self.conv2d(input, kernel_size=(1, 1), filters=filters1, stride=strides)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.conv2d(x, kernel_size=(3, 3), filters=filters2, stride=(1, 1), padding="same")
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.conv2d(x, kernel_size=(1, 1), filters=filters3, stride=(1, 1))
        x = self.batch_normalization(x)

        shortcut = self.conv2d(input, kernel_size=(1, 1), filters=filters3, stride=strides)
        shortcut = self.batch_normalization(shortcut)

        x += shortcut

        return tf.nn.relu(x)

    def id_block(self, input, filters, strides=(1, 1)):
        filters1, filters2, filters3 = filters
        x = self.conv2d(input, kernel_size=(1, 1), filters=filters1, stride=strides)
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.conv2d(x, kernel_size=(3, 3), filters=filters2, stride=strides, padding="same")
        x = self.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.conv2d(x, kernel_size=(1, 1), filters=filters3, stride=strides)
        x = self.batch_normalization(x)

        x += input
        return tf.nn.relu(x)

    def conv2d(self, input, kernel_size, filters, stride, padding="valid"):

        """convolutional layer
        Args:
          input: 4-D tensor [batch_size, height, width, depth]
          kernel_size: [k_height, k_width]
          filters:
          stride:
        Return:
          output: 4-D tensor [batch_size, height/stride, width/stride, out_channels]
        """
        conv1 = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=stride,
        )

        return conv1

    def max_pool(self, input, kernel_size, strides):
        """max_pool layer
        Args:
          input: 4-D tensor [batch_size, height, width, depth]
          kernel_size: [k_height, k_width]
          stride: int32
        Return:
          output: 4-D tensor [batch_size, height/stride, width/stride, depth]
        """
        return tf.layers.max_pooling2d(inputs=input, pool_size=kernel_size, strides=strides)

    def dense(self, input, units, activation, training, dropout=False):

        shape = input.get_shape().as_list()
        if (len(shape) == 4):
            input = tf.reshape(input, [shape[0], shape[1] * shape[2] * shape[3]])
        if activation == "relu":
            dense = tf.layers.dense(inputs=input, units=units, activation=tf.nn.leaky_relu)
        else:
            dense = tf.layers.dense(inputs=input, units=units)

        if dropout == True:
            dropout = tf.layers.dropout(inputs=dense, training=training)
        return dense

    def loss(self, predicts, labelsohe):
        return tf.losses.softmax_cross_entropy(onehot_labels=labelsohe, logits=predicts)

    def batch_normalization(self, input):

        return tf.layers.batch_normalization(input)

    def leaky_relu(self, input, alpha):
        return tf.nn.leaky_relu(input, alpha)
