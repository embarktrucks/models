# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import sys
import traceback
from lxml import etree
import PIL.Image
import tensorflow as tf
import sys
sys.path.insert(0, '.')
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_boolean('clip', True, 'Path to output TFRecord')
FLAGS = flags.FLAGS

from utils import dataset_util
import tensorflow as tf
import numpy as np
import json
import datetime
import cv2
import pytz
from datetime import datetime, timedelta
from multiprocessing import Pool
import os
from sqlalchemy import or_, and_
from embark.services.thedb import model as m
from embark import services
from embark.dqi.sensor_data_interface import SensorDataInterface
from pyspark import SparkContext, SparkConf
services.rds.load(db='production')

SENSOR_DATA = SensorDataInterface()

DATA_PATH = '/sbox/data'
CAMERA = 3


LABEL_TO_INDEX = {
    "background": 0,
    "car": 1,
    "semi": 2,
    "truck": 3,
    "van": 4,
    "motorcycle": 5,
    "other-vehicle": 6
}


# def extract_cuboid_bbox(cuboid_data, image_shape):
#     height, width = image_shape
#     label_text = cuboid_data['label']
#     label = LABEL_TO_INDEX[label_text]
#     ymin = None
#     xmin = None
#     ymax = None
#     xmax = None

#     for v in cuboid_data['vertices']:
#         if v['description'] == 'face-topleft':
#             xmin = int(v['x'])
#             ymin = int(v['y'])

#         if v['description'] == 'face-bottomright':
#             xmax = int(v['x'])
#             ymax = int(v['y'])
#     ymin = int(max(0, ymin)) / float(height)
#     xmin = int(max(0, xmin)) / float(width)
#     ymax = int(min(height, ymax)) / float(height)
#     xmax = int(min(width, xmax)) / float(width)

#     if ymax <= 0 or ymax > height or xmax <= 0 or xmax > width or ymin >= height or ymin < 0 or xmin >= width or xmin < 0:
#         return None, None, None, None, None, None

#     return label, label_text, xmin, ymin, xmax, ymax


def extract_cuboid_bbox(cuboid_data, labeled_image_shape, clip_window=True):
    height, width = labeled_image_shape
    label_text = cuboid_data['label']
    label = LABEL_TO_INDEX[label_text]
    ymin = None
    xmin = None
    ymax = None
    xmax = None

    for v in cuboid_data['vertices']:
        if v['description'] == 'face-topleft':
            xmin = int(v['x'])
            ymin = int(v['y'])

        if v['description'] == 'face-bottomright':
            xmax = int(v['x'])
            ymax = int(v['y'])

    ymin = min(max(0, ymin / float(height)), 1.0) if clip_window else ymin / float(height)
    xmin = min(max(0, xmin / float(width)), 1.0) if clip_window else xmin / float(width)
    ymax = max(min(1., ymax / float(height)), 0.0) if clip_window else ymax / float(height)
    xmax = max(min(1., xmax / float(width)), 0.0) if clip_window else xmax / float(width)

    return label, label_text, xmin, ymin, xmax, ymax


def extract_cuboid_bboxes(cuboids,
                          labeled_image_shape,
                          min_bbox_area=0.0002,
                          clip_window=True):
    labels = []
    labels_text = []
    ymins = []
    xmins = []
    ymaxs = []
    xmaxs = []
    for cuboid_data in cuboids:
        label, label_text, xmin, ymin, xmax, ymax = extract_cuboid_bbox(cuboid_data,
                                                                        labeled_image_shape,
                                                                        clip_window=clip_window)
        if label is None:
            continue

        area = (ymax - ymin) * (xmax - xmin)
        if area < min_bbox_area:
            continue
        # only use vehicle or no vehicle
        labels.append(label)
        labels_text.append(label_text.encode('utf8'))
        ymins.append(ymin)
        xmins.append(xmin)
        ymaxs.append(ymax)
        xmaxs.append(xmax)
    ymins = np.expand_dims(ymins, axis=1)
    xmins = np.expand_dims(xmins, axis=1)
    ymaxs = np.expand_dims(ymaxs, axis=1)
    xmaxs = np.expand_dims(xmaxs, axis=1)
    bboxes = np.concatenate([ymins, xmins, ymaxs, xmaxs], axis=1)
    labels = np.array(labels)
    labels_text = np.array(labels_text)
    return labels, labels_text, bboxes


def crop_image(image, top_left, bot_right, gt_boxes, labels, labels_text, clip_window=True):
    """
    Crops the image with top left to bot right
    """
    cropped = image[top_left[0]:bot_right[0], top_left[1]:bot_right[1]]
    image_shape = np.array(image.shape[:2])
    top_left = top_left / image_shape.astype(np.float32)
    bot_right = bot_right / image_shape.astype(np.float32)
    gt_boxes_area = compute_area(gt_boxes)

    clip_window
    intersect_crop = np.logical_not(np.logical_or.reduce((gt_boxes[:, 2] < top_left[0],  # box bottom most area is above top of crop
                                                          # box right most area is left of crop's
                                                          # left wall
                                                          gt_boxes[:, 3] < top_left[1],
                                                          # box top most area is below bot of crop
                                                          gt_boxes[:, 0] > bot_right[0],
                                                          # box left most area is right of crop's
                                                          # right wall
                                                          gt_boxes[:, 1] > bot_right[1])))

    if clip_window:
        intersect_crop = np.ones_like(intersect_crop, dtype=np.bool)
    gt_top_left = np.maximum(gt_boxes[:, :2], np.array([top_left[0], top_left[1]]))
    gt_bot_right = np.minimum(gt_boxes[:, 2:], np.array([bot_right[0], bot_right[1]]))

    cropped_gt_boxes = np.concatenate([gt_top_left, gt_bot_right], axis=1)
    croppped_gt_boxes_area = compute_area(cropped_gt_boxes)
    croppped_gt_boxes_width = compute_width(cropped_gt_boxes)
    croppped_gt_boxes_height = compute_height(cropped_gt_boxes)

    gt_boxes_fraction_orig_area = croppped_gt_boxes_area / gt_boxes_area

    large_portion_in_crop = np.logical_and.reduce((croppped_gt_boxes_width > 0,
                                                   croppped_gt_boxes_height > 0,
                                                   gt_boxes_fraction_orig_area > .4))

    valid_idxs = np.where(np.logical_and(large_portion_in_crop, intersect_crop))

    final_cropped_gt_boxes = gt_boxes[valid_idxs[0]]
    cropped_labels = labels[valid_idxs[0]]
    cropped_labels_text = labels_text[valid_idxs[0]]

    crop_height = (bot_right[0] - top_left[0])
    crop_width = (bot_right[1] - top_left[1])

    cropped_gt_adjusted = final_cropped_gt_boxes - np.concatenate([top_left, top_left]).transpose()
    cropped_gt_adjusted = cropped_gt_adjusted / \
        np.array([crop_height, crop_width, crop_height, crop_width], dtype=np.float32)
    return cropped, cropped_gt_adjusted, cropped_labels, cropped_labels_text


def create_example(image,
                   image_file,
                   bboxes,
                   labels_text,
                   labels,
                   crop):

    encoded_jpg = cv2.imencode('.jpg', image)[1].tostring()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    truncated = [0] * len(labels)
    difficult_obj = [0] * len(labels)
    poses = ['Unspecified'] * len(labels)
    width, height = image.size
    ymins = bboxes[:, 0]
    xmins = bboxes[:, 1]
    ymaxs = bboxes[:, 2]
    xmaxs = bboxes[:, 3]
    feature_set = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/crop_tl_x': dataset_util.int64_feature(crop[0][1]),
        'image/crop_tl_y': dataset_util.int64_feature(crop[0][0]),
        'image/crop_br_x': dataset_util.int64_feature(crop[1][1]),
        'image/crop_br_y': dataset_util.int64_feature(crop[1][0]),
        'image/filename': dataset_util.bytes_feature(image_file.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_file.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(labels_text),
        'image/object/class/label': dataset_util.int64_list_feature(labels),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_set))
    return example


def process_frame_object(frame, clip_window=True):
    image_file = SENSOR_DATA.get_image(frame.truck, frame.timestamp, frame.cam)
    if not os.path.exists(image_file):
        print('Image file does not exist: ', image_file)
        return []
    # hires_file = image_file.replace('images', 'images_hires')
    # if not os.path.exists(hires_file):
    #     # print('Image file does not exist: ', hires_file)
    #     return []

    image = cv2.imread(image_file)
    if image is None:
        # print('Image file is None: ', image_file)
        return []
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cuboid_data = json.loads(frame.data)

    labeled_image_height = 384
    labeled_image_width = 672
    image = cv2.resize(image, (labeled_image_width, labeled_image_height),
                       interpolation=cv2.INTER_LINEAR)
    labeled_image_shape = [labeled_image_height, labeled_image_width]
    labels, labels_text, boxes = extract_cuboid_bboxes(cuboid_data,
                                                       labeled_image_shape,
                                                       clip_window=clip_window)
    num_cuboids = boxes.shape[0]
    # if num_cuboids == 0:
    #     print('no cuboids found for: ', frame.timestamp, frame)
    #     print(image_file)
    #     return []
    #

    image_shape = image.shape
    factor = 5.
    horizon = 80
    side_buffer = 80
    crop_boxes = [(np.array([0, 0], dtype=np.int32), np.array(labeled_image_shape, dtype=np.int32))]

    # tabbed out for ow res images
    #crop_boxes = [(np.array([0, 0], dtype=np.int32), np.array([728, 1288], dtype=np.int32))]
    # for r in [horizon]:
    #     for c in np.linspace(side_buffer + image_shape[1] / factor,
    #                          -side_buffer + (factor - 2) * image_shape[1] / factor, num=3):
    #         top_left = np.array([r, c], dtype=np.int32)
    #         bot_right = np.array([r + image_shape[0] / factor,
    #                               c + image_shape[1] / factor], dtype=np.int32)
    #         crop_boxes.append((top_left, bot_right))
    examples = []
    for top_left, bot_right in crop_boxes:
        cropped, cropped_gt_adjusted, cropped_labels, cropped_labels_text = crop_image(image,
                                                                                       top_left,
                                                                                       bot_right,
                                                                                       boxes,
                                                                                       labels,
                                                                                       labels_text)
        if cropped_gt_adjusted.shape[0] == 0:
            continue
        example = create_example(cropped,
                                 image_file,
                                 cropped_gt_adjusted,
                                 cropped_labels_text,
                                 cropped_labels,
                                 (top_left, bot_right))
        examples.append(example)
    return examples


def get_frame_objects():

    frame_filter = and_(m.FrameLabels.data != None,
                        m.FrameLabels.label_type == m.FrameLabels.LabelTypes.VEHICLE_CUBOID,
                        or_(m.FrameLabels.cam == 4, m.FrameLabels.cam == 3))
    try:
        db = services.rds.scoped_session()
        frame_objects = db.query(m.FrameLabels)\
            .filter(frame_filter)\
            .order_by(-m.FrameLabels.timestamp)\
            .limit(100)\
            .all()
    finally:
        db.close()

    return frame_objects


def save_dataset(sc, save_path, frame_objects, frames_per_record=500, num_workers=48, clip_window=True):
    frame_objects = list(frame_objects)
    import random
    random.shuffle(frame_objects)

    try:
        import os
        os.makedirs(save_path)
    except:
        pass

    num_partitions = int(np.ceil(len(frame_objects) / float(frames_per_record)))
    partitions = [(i, []) for i in range(num_partitions)]

    for i, frame in enumerate(frame_objects):
        partition = i % num_partitions
        partitions[partition][1].append(frame)

    def save_partition(partition, frames):
        record_file = save_path + '/' + str(partition) + '.tf'
        writer = tf.python_io.TFRecordWriter(record_file)
        num_frames = 0
        num_examples = 0
        for i, frame in enumerate(frames):
            if i % 50 == 0:
                print("Partition:", partition, "Frame: ", i, 'Found Frames:',
                      num_frames, 'Num examples', num_examples)
            try:
                examples = process_frame_object(frame, clip_window=clip_window)
            except:
                print("Exception with: ", frame.timestamp, traceback.format_exc())
                continue

            if len(examples) == 0:
                continue
            num_frames += 1
            for example in examples:
                writer.write(example.SerializeToString())
                num_examples += 1
        writer.close()
        return num_examples, num_frames

    if num_workers == 1:
        counts = [save_partition(partition, frames) for partition, frames in partitions]
    else:
        counts = sc.parallelize(partitions, num_workers)\
            .map(lambda (partition, frames): save_partition(partition, frames))\
            .collect()

    print('Num Frames in Dataset:', np.sum(counts, axis=0))


def compute_area(boxes):
    h = compute_height(boxes)
    w = compute_width(boxes)
    pos = np.logical_and(h > 0, w > 0)
    return h * w * pos


def compute_height(boxes):
    return boxes[:, 2] - boxes[:, 0]


def compute_width(boxes):
    return boxes[:, 3] - boxes[:, 1]


def compute_diag(boxes):
    return np.sqrt(np.power(compute_height(boxes), 2) + np.power(compute_width(boxes), 2))


def main(_):

    # dataset_file = FLAGS.dataset_file

    # datetimes = []
    # with open(dataset_file) as f:
    #     for line in f:
    #         line = line.strip()
    #         digits = [int(d) for d in line.split('/')]
    #         dt = pytz.utc.localize(datetime(*digits))
    #         datetimes.append(dt)

    import random

    frame_objects = get_frame_objects()
    cam_frames = {}
    for frame in frame_objects:
        if frame.cam not in cam_frames:
            cam_frames[frame.cam] = []
        cam_frames[frame.cam].append(frame)

    # we are testing cam4
    num_total = len(cam_frames[4])
    num_test = int(num_total * .05)

    train_set = cam_frames[4][num_test:]
    test_set = cam_frames.get(3,[]) + cam_frames[4][:num_test]

    print("TRAIN FRAMES: ", len(train_set))
    print("TEST FRAMES: ", len(test_set))

    num_workers = 1
    conf = SparkConf()
    conf.setMaster('local[*]')
    conf.setAppName('emabark_records')
    conf.set("spark.executor.instances", str(num_workers))
    conf.set("spark.driver.maxResultSize", "15g")
    conf.set("spark.executor.cores", "1")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.driver.memory", "12g")
    conf.set("spark.python.worker.memory", "4g")

    sc = SparkContext(conf=conf)

    print("saving train")
    save_dataset(sc, FLAGS.output_path + '/train', train_set,
                 num_workers=num_workers, clip_window=FLAGS.clip)
    print("saving test")
    save_dataset(sc, FLAGS.output_path + '/test', test_set,
                 num_workers=num_workers, clip_window=FLAGS.clip)

if __name__ == '__main__':
    tf.app.run()
