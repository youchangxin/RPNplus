# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import utils.tools as tools

from configuration import cfg

class Dataset(object):
    def __init__(self, dataset_type):
        self.TFRECORD_DIR       = cfg.TRAIN.TFRECORD_DIR  if dataset_type == 'train' else cfg.TEST.TFRECORD_DIR
        self.batch_size         = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        #self.TXT_DIR            = cfg.TRAIN.TXT_DIR if dataset_type == 'train' else cfg.TEST.TXT_DIR

        self.input_sizes        = cfg.RPN.INPUT_SIZE
        self.grid               = cfg.RPN.STRIDE
        self.anchors            = np.array(cfg.RPN.ANCHORS)
        self.max_bbx_per_img    = cfg.RPN.MAX_PER_IMAGE
        self.pos_thresh         = cfg.RPN.POS_THRESH
        self.neg_thresh         = cfg.RPN.NEG_THRESH

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

        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
        feature_dict['image'] = tf.image.resize(feature_dict['image'], self.input_sizes)


        return feature_dict['image'], feature_dict['bbox']

    def __preprocess_data(self, images, bboxes):
        images = images.numpy()
        bboxes = bboxes.numpy()

        images_norm = images / 255.0
        fixed_bboxes = np.zeros(shape=[self.max_bbx_per_img, 4])
        if bboxes.shape[0] < self.max_bbx_per_img:
            for i in range(bboxes.shape[0]):
                fixed_bboxes[i] = bboxes[i]
        else:
            fixed_bboxes[:, :] = bboxes[ :self.max_bbx_per_img, :]

        return images_norm, fixed_bboxes

    def __encode_label(self, gt_boxes):
        target_scores = np.zeros(shape=[45, 60, 9, 2])  # 0: background, 1: foreground, ,
        target_bboxes = np.zeros(shape=[45, 60, 9, 4])  # t_x, t_y, t_w, t_h
        target_masks = np.zeros(shape=[45, 60, 9])  # negative_samples: -1, positive_samples: 1

        valid_bbx_index = tools.get_valid_num_bboxes(gt_boxes)
        gt_boxes = gt_boxes[:valid_bbx_index, :]

        for i in range(45): # height
            for j in range(60): # width
                for k in range(9): # anchor scales
                    # generate grid
                    center_x = j * self.grid + self.grid * 0.5
                    center_y = i * self.grid + self.grid * 0.5

                    xmin = center_x - self.anchors[k][0] * 0.5
                    ymin = center_y - self.anchors[k][1] * 0.5
                    xmax = center_x + self.anchors[k][0] * 0.5
                    ymax = center_y + self.anchors[k][1] * 0.5

                    # filter some invalid defauld box that out of prescribed range
                    if (xmin > -5) & (ymin > -5) & (xmax < (self.input_sizes[1] + 5)) & (ymax < (self.input_sizes[0] + 5)):
                        anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                        anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                        # compute iou between this anchor boxes and all ground-truth boxes
                        ious = tools.iou(anchor_boxes, gt_boxes)
                        positive_masks = ious >= self.pos_thresh
                        negative_masks = ious <= self.neg_thresh

                        if np.any(positive_masks):
                            target_scores[i, j, k, 1] = 1.
                            target_masks[i, j, k] = 1 # labeled as a positive sample
                            # find out which ground-truth box matches this anchor
                            max_iou_index = np.argmax(ious)
                            selected_gt_boxes = gt_boxes[max_iou_index]
                            target_bboxes[i, j, k] = tools.compute_regression(selected_gt_boxes, anchor_boxes[0])

                        if np.all(negative_masks):
                            target_scores[i, j, k, 0] = 1.
                            target_masks[i , j, k] = -1  #labeled as a negative sample
        return target_scores, target_bboxes, target_masks

    def __generate_target(self, images, bboxes):
        bboxes = bboxes.numpy()

        target_scores = np.zeros(shape=[self.batch_size, 45, 60, 9, 2], dtype=np.float)
        target_bboxes = np.zeros(shape=[self.batch_size, 45, 60, 9, 4], dtype=np.float)
        target_masks = np.zeros(shape=[self.batch_size,45, 60, 9], dtype=np.int)
        for i in range(self.batch_size):
            target = self.__encode_label(bboxes[i])
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]

        return images, target_scores, target_bboxes, target_masks


    def genertate_dataset(self):
        dataset = tf.data.TFRecordDataset(self.TFRECORD_DIR)
        dataset = dataset.map(map_func=self.__parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(map_func=lambda images, bboxes: tf.py_function(self.__preprocess_data,
                                                                             inp=[images, bboxes],
                                                                             Tout=[tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.map(map_func=lambda images, gt_boxes: tf.py_function(self.__generate_target,
                                                                               inp=[images, gt_boxes],
                                                                               Tout=[tf.float32,tf.float32,
                                                                                     tf.float32,tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = Dataset('train').genertate_dataset()
    for image, target_scores, target_bboxes, target_masks in dataset:
        print(target_bboxes.numpy().shape)

        image = image[0]

        #image = tf.cast(image, tf.int32)

        plt.imshow(image.numpy())
        plt.show()