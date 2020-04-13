# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import utils.postprocess as preprocess

from configuration import cfg

class Dataset(object):
    def __init__(self, dataset_type):
        self.TFRECORD_DIR       = cfg.TRAIN.TFRECORD_DIR  if dataset_type == 'train' else cfg.TEST.TFRECORD_DIR
        self.batch_size         = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        #self.TXT_DIR            = cfg.TRAIN.TXT_DIR if dataset_type == 'train' else cfg.TEST.TXT_DIR

        self.input_sizes        = cfg.RPN.INPUT_SIZE

    def __parse_example(self, example_string):
        # define structure of Feature
        feature_description = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'bbox': tf.io.VarLenFeature(dtype=tf.int64),
        }
        # decode the TFRecord file
        feature_dict = tf.io.parse_single_example(example_string, feature_description)

        # transite SparseTensor to DenseTensor
        #feature_dict['image'] = tf.sparse.to_dense(feature_dict['image'])
        feature_dict['bbox'] = tf.sparse.to_dense(feature_dict['bbox'])
        # reshape tensor
        #feature_dict['image'] = tf.reshape(feature_dict['image'], [])
        feature_dict['bbox'] = tf.reshape(feature_dict['bbox'], [-1, 4])

        print(feature_dict['bbox'])
        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
        feature_dict['image'] = tf.image.resize(feature_dict['image'], self.input_sizes)
        print(feature_dict['image'])

        return feature_dict['image'], feature_dict['bbox']

    def genertate_dataset(self):
        dataset = tf.data.TFRecordDataset(self.TFRECORD_DIR)
        dataset = dataset.map(map_func=self.__parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = Dataset('train').genertate_dataset()
    for image, bbox in dataset:
        print(bbox.numpy)
        print(image.shape)

        image = tf.squeeze(image)
        image = tf.cast(image, tf.float32)

        plt.imshow(image.numpy())
        plt.show()