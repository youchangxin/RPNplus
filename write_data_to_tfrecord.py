# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

from configuration import cfg


class ParseDate(object):
    def __init__(self):
        self.image_dir = './data/synthetic_dataset/image'
        self.annot_dir = './data/synthetic_dataset/imageAno'

    def __combine_info(self, image_name):
        image_path = self.image_dir + '/' + image_name
        return image_path

    def __parse_txt(self, txt_name):
        image_name = txt_name.split('.')[0] + '.jpg'
        image_path = self.__combine_info(image_name)
        txt = os.path.join(self.annot_dir, txt_name)
        bboxes = open(txt).readlines()[1:]
        roi = np.zeros([len(bboxes), 4], dtype=np.int64)
        for iter_, bb in zip(range(len(bboxes)), bboxes):
            bb = bb.replace('\n', '').split(' ')
            bbtype = bb[0]
            bba = np.array([int(bb[i]) for i in range(1, 5)])
            # occ = float(bb[5])
            # bbv = np.array([float(bb[i]) for i in range(6, 10)])
            #ignore = int(bb[10])

            #ignore = ignore or (bbtype != 'person')
            #ignore = ignore or (bba[3] < 40)
            bba[2] += bba[0]
            bba[3] += bba[1]

            roi[iter_, :4] = bba
        roi = roi.flatten()
        roi = roi.tolist()
        return image_path, roi

    def write_tfrecoed(self, tfrecord_path):
        images = []
        bboxes = []
        for item in os.listdir(self.annot_dir):
            image, bbox = self.__parse_txt(item)
            images.append(image)
            bboxes.append(bbox)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            count = 1
            for image, bbox in zip(images, bboxes):
                print('Writing picture of {}'.format(count))
                count += 1

                image = open(image,mode='rb').read()
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


ParseDate().write_tfrecoed(cfg.TRAIN.TFRECORD_DIR)
