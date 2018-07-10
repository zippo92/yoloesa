import tensorflow as tf


class Yolo():

    def __init__(self):
        return

    def inference(self, images, mode):
        import ConfigParser

        config = ConfigParser.ConfigParser()
        config.read("../config/layers.cfg")
        predicts = images
        for layer in config.sections():

            if layer.startswith("Conv"):
                kernel = config.get(layer, "kernel_size")
                filters = config.get(layer, "filters")
                stride = config.get(layer, "stride")
                predicts = self.conv2d(predicts,kernel,filters,stride)

            if layer.startswith("MaxPool"):
                kernel = config.get(layer, "kernel_size")
                stride = config.get(layer, "stride")
                predicts = self.max_pool(predicts,kernel,stride)

            if layer.startswith("DoubleConv"):
                kernel_1 = config.get(layer, "kernel_size_1")
                filters_1 = config.get(layer, "filters_1")
                stride_1 = config.get(layer, "stride_1")
                kernel_2 = config.get(layer, "kernel_size_2")
                filters_2 = config.get(layer, "filters_2")
                stride_2 = config.get(layer, "stride_2")

                for i in range(config.getint(layer, "repeat")):
                    predicts = self.conv2d(predicts, kernel_1, filters_1, stride_1)
                    predicts = self.conv2d(predicts,kernel_2,filters_2,stride_2)

            if layer.startswith("Fully"):
                units = config.get(layer, "units")
                dropOutRate = config.get(layer, "dropOutRate")
                predicts = self.dense(predicts, units, dropOutRate)

    #    TODO MANCA ULTIMO STRATO

        # # Logits Layer
        # logits = tf.layers.dense(inputs=dropout, units=10)
        #
        # predictions = {
        #     # Generate predictions (for PREDICT and EVAL mode)
        #     "classes": tf.argmax(input=logits, axis=1),
        #     # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        #     # `logging_hook`.
        #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # }

        return predicts


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

    def dense(self, input, units, dropoutrate,mode):

        reshape = tf.reshape(input, [tf.shape(input)[0], -1])
        dense = tf.layers.dense(inputs=reshape, units=units, activation=tf.nn.relu)

        dropout = tf.layers.dropout(
            inputs=dense, rate=dropoutrate, training=mode == tf.estimator.ModeKeys.TRAIN)
        return dropout


    def loss(self, predicts, labels):
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predicts)

    def construct_graph(self):
        # construct graph
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

        self.predicts = self.net.inference(self.images)
        self.total_loss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objects_num)

        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self._train()



if __name__ == '__main__':
    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.read("../config/layers.cfg")

    for layer in config.sections():

        if layer.startswith("Conv"):
            print layer
            kernel = config.get(layer, "kernel_size")
            filters = config.get(layer, "filters")
            stride = config.get(layer, "stride")
            print kernel
            print filters
            print stride
        if layer.startswith("MaxPool"):
            print layer
            kernel = config.get(layer, "kernel_size")
            stride = config.get(layer, "stride")
        if layer.startswith("DoubleConv"):
            kernel_1 = config.get(layer, "kernel_size_1")
            filters_1 = config.get(layer, "filters_1")
            stride_1 = config.get(layer, "stride_1")
            kernel_2 = config.get(layer, "kernel_size_2")
            filters_2 = config.get(layer, "filters_2")
            stride_2 = config.get(layer, "stride_2")
            for i in range(config.getint(layer, "repeat")):
                print i
                print kernel_1
                print filters_1
                print stride_1
                print kernel_2
                print filters_2
                print stride_2
        if layer.startswith("Fully"):
            units = config.get(layer, "units")
            dropOutRate = config.get(layer, "dropOutRate")
            print units
            print dropOutRate




