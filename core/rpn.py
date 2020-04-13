# -*- coding: utf-8 -*-
import tensorflow as tf

class RPNplus(tf.keras.Model):
    def __init__(self):
        super(RPNplus, self).__init__()
        # VGG-16 layer1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                               strides=2,
                                               padding='same')
        # VGG-16 layer2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                               strides=2,
                                               padding='same')
        # VGG-16 layer3
        self.conv3_1 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv3_3 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                               strides=2,
                                               padding='same')
        # VGG-16 layer4
        self.conv4_1 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv4_3 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                               strides=2,
                                               padding='same')
        # VGG-16 layer5
        self.conv5_1 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv5_2 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        self.conv5_3 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=[3, 3],
                                              activation='relu',
                                              padding='same')
        # Region proposal convolution
        self.max_pool_p3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                                     strides=2,
                                                     padding='same')
        self.rpn_conv1 = tf.keras.layers.Conv2D(filters=256,
                                                kernel_size=[5, 2],
                                                activation='relu',
                                                padding='same')
        self.rpn_conv2 = tf.keras.layers.Conv2D(filters=256,
                                                kernel_size=[5, 2],
                                                activation='relu',
                                                padding='same')
        self.rpn_conv3 = tf.keras.layers.Conv2D(filters=256,
                                                kernel_size=[5, 2],
                                                activation='relu',
                                                padding='same')
        # Bounding Boxes Regression layer
        self.bboxes_conv = tf.keras.layers.Conv2D(filters=36,
                                                  kernel_size=[1, 1],
                                                  padding='same',
                                                  use_bias=False)
        # Scores layer
        self.scores_conv = tf.keras.layers.Conv2D(filters=18,
                                                  kernel_size=[1, 1],
                                                  padding='same',
                                                  use_bias=False)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1_1(inputs, training=training)
        x = self.conv1_2(x, training=training)
        x = self.pool1(x)

        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.pool2(x)

        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.pool3(x)

        # Branch1
        pool3_p = self.max_pool_p3(x)
        pool3_p = self.rpn_conv1(pool3_p, training=training)  # [1, 45, 60, 256]

        # Branch2
        x = self.conv4_1(x, training=training)
        x = self.conv4_2(x, training=training)
        x = self.conv4_3(x, training=training)
        x = self.pool4(x)
        pool4_p = self.rpn_conv2(x, training=training)  # [1, 45, 60, 512]

        # Branch3
        x = self.conv5_1(x, training=training)
        x = self.conv5_2(x, training=training)
        x = self.conv5_3(x, training=training)
        pool5_p = self.rpn_conv3(x, training=training)  # [1, 45, 60, 512]

        region_proposal = tf.concat([pool3_p, pool4_p, pool5_p], axis=-1)  # [1, 45, 60, 1280]

        conv_cls_score = self.scores_conv(region_proposal, training=training)
        conv_cls_bboxes = self.bboxes_conv(region_proposal, training=training)

        cls_scores = tf.reshape(conv_cls_score, [-1, 45, 60, 9, 2])
        cls_bboxes = tf.reshape(conv_cls_bboxes, [-1, 45, 60, 9, 4])

        return cls_scores, cls_bboxes
