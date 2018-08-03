import tensorflow as tf


class Yolo2():

    def __init__(self):
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    def inference(self, images, reuse = False):

        with tf.variable_scope('yolo', reuse = reuse):

            # Layer 1
            predicts = self.conv2d(input = images,kernel_size=3,filters=32,stride=1)
            predicts = self.batch_normalization(input = predicts)
            predicts = self.leaky_relu(predicts, 0.1)
            predicts = self.max_pool(input=predicts,kernel_size=2,strides=2)

            # Layer 2
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=64, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)
            predicts = self.max_pool(input=predicts, kernel_size=2, strides=2)

            # Layer 3
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=128, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 4
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=64, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 5
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=128, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)
            predicts = self.max_pool(input=predicts, kernel_size=2, strides=2)

            # Layer 6
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=256, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 7
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=128, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 8
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=256, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)
            predicts = self.max_pool(input=predicts, kernel_size=2, strides=2)

            # Layer 9
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=512, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 10
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=256, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 11
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=512, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 12
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=256, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 13
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=512, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            skip = predicts

            predicts = self.max_pool(input=predicts, kernel_size=2, strides=2)


            # Layer 14
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=1024, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 15
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=512, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 16
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=1024, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 17
            predicts = self.conv2d(input=predicts, kernel_size=1, filters=512, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 18
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=1024, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            # Layer 21
            skip = self.conv2d(input= skip, kernel_size=1, filters=64, stride=1)
            skip = self.batch_normalization(skip)
            skip = self.leaky_relu(skip,0.1)
            skip = tf.space_to_depth(skip,block_size=2)

            predicts = tf.concat([skip, predicts],axis = 3)

            # Layer 22
            predicts = self.conv2d(input=predicts, kernel_size=3, filters=1024, stride=1)
            predicts = self.batch_normalization(input=predicts)
            predicts = self.leaky_relu(predicts, 0.1)

            output = self.convert_to_bbox(predicts)

            return output

    def convert_to_bbox(self, x):
        """Convert final layer features to bounding box parameters.
        Returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution.
        box_conf : tensor
            Probability estimate for whether each box contains any object.
        box_class_pred : tensor
            Probability distribution estimate for each box over class labels.
        """
        with tf.variable_scope('convert_to_bbox'):
            num_anchors = self.anchors.shape[0]
            anchors = tf.keras.backend.variable(self.anchors)
            # Reshape to batch, height, width, num_anchors, box_params.
            anchors = tf.reshape(anchors, [1, 1, 1, num_anchors, 2])

            # Dynamic implementation of conv dims for fully convolutional model.
            conv_dims = x.get_shape().as_list()[1:3]
            conv_height_index = tf.keras.backend.arange(0, stop=conv_dims[0])
            conv_width_index = tf.keras.backend.arange(0, stop=conv_dims[1])
            # In YOLO the height index is the inner most iteration.

            conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])
            conv_width_index = tf.tile(
                tf.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
            conv_width_index = tf.keras.backend.flatten(tf.transpose(conv_width_index))
            conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
            conv_index = tf.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
            conv_index = tf.cast(conv_index, x.dtype)

            x = tf.reshape(
                x, [-1, conv_dims[0], conv_dims[1], num_anchors, self.num_classes + 5])
            conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), x.dtype)

            box_xy = tf.sigmoid(x[..., :2])
            box_wh = tf.exp(x[..., 2:4])
            box_confidence = tf.sigmoid(x[..., 4:5])
            box_class_probs = tf.nn.softmax(x[..., 5:])

            box_xy = (box_xy + conv_index) / conv_dims
            box_wh = box_wh * anchors / conv_dims

            x[..., :2] =  box_xy
            x[..., 2:4] = box_wh
            x[..., 4] = box_confidence
            x[..., 5:] = box_class_probs

            return x

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
            activation=None,
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

    def dense(self, input, units, dropoutrate, activation):


        shape = input.get_shape().as_list()
        if(len(shape) == 4):
            input = tf.reshape(input, [shape[0], shape[1]*shape[2]*shape[3]])
        if activation == "relu":
            dense = tf.layers.dense(inputs=input, units=units, activation=tf.nn.leaky_relu)
        else:
            dense = tf.layers.dense(inputs=input, units=units)


        # dropout = tf.layers.dropout(
        #    inputs=dense, rate=dropoutrate, training=True)
        return dense

    def batch_normalization(self, input):

        return tf.layers.batch_normalization(input)

    def leaky_relu(self, input, alpha):
        return tf.nn.leaky_relu(input, alpha)

    """
    matching_true_boxes: [bb,gridx, gridy,anchors, [xcenter,ycenter,hcenter,wcenter,1,label_ohe]]
    detector_mask: [[bb,gridx, gridy,anchors,1]
    predicts: [bb, box_xy, box_wh, box_confidence, box_class_probs]
    """
    def loss(self, predicts, matching_true_boxes, detector_mask):


        ### adjust confidence
        true_wh_half = matching_true_boxes[...,2:4] / 2.
        true_mins = matching_true_boxes[...,0:2] - true_wh_half
        true_maxes = matching_true_boxes[...,0:2] + true_wh_half

        pred_wh_half = predicts[...,2:4] / 2.
        pred_mins = predicts[...,0:2]- pred_wh_half
        pred_maxes = predicts[...,0:2] + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = matching_true_boxes[...,2] * matching_true_boxes[...,3]
        pred_areas = predicts[...,2] * predicts[...,3]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * matching_true_boxes[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(matching_true_boxes[..., 5:], -1)



        class_mask_updates = tf.ones(shape=(tf.shape(detector_mask)[0]))
        class_mask_indices = tf.cast(tf.where(detector_mask > 0.), dtype=tf.int32)
        class_mask = tf.scatter_nd(class_mask_indices[:, 0:2], class_mask_updates, tf.shape(detector_mask)[:,0:2])

        noobj_detector_mask = tf.bitwise.invert(tf.cast(detector_mask, dtype=tf.int32)) + 2



        nb_coord_box = tf.reduce_sum(tf.to_float(detector_mask > 0.))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.))
        nb_noobj_box = tf.reduce_sum(tf.to_float(class_mask > 0))

        loss_x = tf.reduce_sum(tf.square(matching_true_boxes[...,0] - predicts[...,0]) * detector_mask) / (nb_coord_box) / 2.
        loss_y = tf.reduce_sum(tf.square(matching_true_boxes[...,1] - predicts[...,1]) * detector_mask) / (nb_coord_box) / 2.
        loss_h = tf.reduce_sum(tf.square(matching_true_boxes[...,2] - predicts[...,2]) * detector_mask) / (nb_coord_box) / 2.
        loss_w = tf.reduce_sum(tf.square(matching_true_boxes[...,3] - predicts[...,3]) * detector_mask) / (nb_coord_box) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - matching_true_boxes[...,4]) * detector_mask) / (nb_class_box) / 2.
        loss_noobj_conf = tf.reduce_sum(tf.square(true_box_conf - matching_true_boxes[...,4]) * noobj_detector_mask) / (nb_noobj_box) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=predicts[...,5:])
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box)

        loss = self.lambda_coord *(loss_x + loss_y + loss_h + loss_w + loss_conf) + self.lambda_noobj * loss_noobj_conf + loss_class

        return loss

