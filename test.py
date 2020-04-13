# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from core.rpn import RPNplus
from utils.postprocess import decode_output, plot_boxes_on_image, nms

dataset_path ="./data/synthetic_dataset"
prediction_result_path = "./prediction"
if not os.path.exists(prediction_result_path): os.mkdir(prediction_result_path)

model = RPNplus()
model.load_weights("./RPN.h5")

for ind in range(8000, 8200):
    image_path = os.path.join(dataset_path, "image/%d.jpg" %(ind+1))
    raw_img = cv2.imread(image_path)
    image_data = np.expand_dims(raw_img / 255.0, 0)

    pred_scores, pred_bboxes = model(image_data)
    pred_scores = tf.nn.softmax(pred_scores, axis=-1)
    pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores, 0.9)
    pred_bboxes = nms(pred_bboxes, pred_scores, 0.5)

    plot_boxes_on_image(raw_img, pred_bboxes)

    save_path = os.path.join(prediction_result_path, str(ind) + ".jpg")
    print("=> saving prediction results into %s" % save_path)
    Image.fromarray(raw_img).save(save_path)
