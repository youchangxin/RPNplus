# -*- coding: utf-8 -*-
import os
import cv2
import random
import tensorflow as tf
import shutil
import numpy as np
import utils.tools as tools

from dataset import Dataset
from core.rpn import RPNplus
from configuration import cfg
from core.loss import Loss


logdir = "./log"
EPOCH = cfg.TRAIN.EPOCH
save_frequency = cfg.TRAIN.SAVE_FREQ
image_height, image_width = cfg.RPN.INPUT_SIZE
lambda_scale = cfg.TRAIN.LAMBDA_SCALE

# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
trainset = Dataset('train')

model = RPNplus()
model.build(input_shape=(None, image_height, image_width, 3))
model.summary()

# TensorBoard
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

optimizer = tf.keras.optimizers.Adam(lr=cfg.TRAIN.LR)
loss = Loss()


def train_step(images, target_scores, target_bboxes, target_masks, epoch):
    with tf.GradientTape() as tape:
        pred_scores, pred_bboxes = model(images, training=True)

        # computing Loss
        score_loss, boxes_loss = loss.compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes)
        total_loss = score_loss + lambda_scale * boxes_loss

        # Gradient
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f" % (epoch, global_steps,
                                                                                              total_loss.numpy(),
                                                                                              score_loss.numpy(),
                                                                                              boxes_loss.numpy()))
        global_steps.assign_add(1)

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=global_steps)
            tf.summary.scalar("score_loss", score_loss, step=global_steps)
            tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
        writer.flush()


if __name__ == "__main__":
    for epoch in range(cfg.TRAIN.EPOCH):

        for image, target_scores, target_bboxes, target_masks in trainset.genertate_dataset():
            train_step(image, target_scores, target_bboxes, target_masks, epoch)

            # save model weights
        if epoch % save_frequency == 0:
            model.save_weights(filepath=cfg.RPN.SAVE_MODEL_DIR + "RPNplus_epoch-{}.h5".format(epoch))

    model.save_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "RPN.h5")

