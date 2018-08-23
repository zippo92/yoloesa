import tensorflow as tf


class Net():

    def __init__(self):
        pass
    def inference(self, images,training = True,reuse = False):
        import ConfigParser

        config = ConfigParser.ConfigParser()
        config.read("config/vgg16.cfg")
        predicts = images
        with tf.variable_scope('vgg16', reuse = reuse):
            for layer in config.sections():

                if layer.startswith("Conv"):
                    kernel = int(config.get(layer, "kernel_size"))
                    filters = int(config.get(layer, "filters"))
                    stride = int(config.get(layer, "stride"))
                    predicts = self.conv2d(predicts,kernel,filters,stride)

                if layer.startswith("MaxPool"):
                    kernel = int(config.get(layer, "kernel_size"))
                    stride = int(config.get(layer, "stride"))
                    predicts = self.max_pool(predicts,kernel,stride)

                if layer.startswith("DoubleConv"):
                    kernel_1 = int(config.get(layer, "kernel_size_1"))
                    filters_1 = int(config.get(layer, "filters_1"))
                    stride_1 = int(config.get(layer, "stride_1"))
                    kernel_2 = int(config.get(layer, "kernel_size_2"))
                    filters_2 = int(config.get(layer, "filters_2"))
                    stride_2 = int(config.get(layer, "stride_2"))

                    for i in range(config.getint(layer, "repeat")):
                        predicts = self.conv2d(predicts, kernel_1, filters_1, stride_1)
                        predicts = self.conv2d(predicts,kernel_2,filters_2,stride_2)

                if layer.startswith("Fully"):
                    units = int(config.get(layer, "units"))
                    dropOutRate = float(config.get(layer, "dropOutRate"))
                    activation = config.get(layer, "activation")
                    predicts = self.dense(predicts, units, dropOutRate, activation,training = training)

            # logits = tf.nn.softmax(predicts)

            return predicts#, logits


    def conv2d(self, input, kernel_size,filters, stride):


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
            padding="same",
            activation=tf.nn.relu,
            strides = stride,
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

    def dense(self, input, units, dropoutrate, activation, training):


        shape = input.get_shape().as_list()
        if(len(shape) == 4):
            input = tf.reshape(input, [shape[0], shape[1]*shape[2]*shape[3]])
        if activation == "relu":
            dense = tf.layers.dense(inputs=input, units=units, activation=tf.nn.leaky_relu)
        else:
            dense = tf.layers.dense(inputs=input, units=units)

        dense = tf.layers.dropout(inputs=dense, rate=dropoutrate, training=training)
        return dense


    def loss(self, predicts, labelsohe):
        return tf.losses.softmax_cross_entropy(onehot_labels=labelsohe, logits=predicts)
