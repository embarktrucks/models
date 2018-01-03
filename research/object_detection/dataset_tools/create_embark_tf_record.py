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

from lxml import etree
import PIL.Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('dataset_file', '', 'file for dataset.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

from utils import dataset_util
import tensorflow as tf
import numpy as np
import json
import datetime
import pytz
from datetime import datetime, timedelta
from multiprocessing import Pool
import os
from sqlalchemy import or_, and_
from embark.dqi.sensor_data_interface import SensorDataInterface
from embark.services import RDSService
import embark.services.thedb.model as m
from embark.perception.objects.training.data_providers.ssd import SSD

FrameObjects = m.FrameObjects
SENSOR_DATA = SensorDataInterface('/sbox/data')

DATA_PATH = '/sbox/data'
CAMERA = 3


LABEL_TO_INDEX = {
    "background": 0,
    "truck": 1,
    "semi": 2,
    "car": 3,
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


def extract_cuboid_bbox(cuboid_data, image_shape):
    height, width = image_shape
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
    ymin = min(max(0, ymin/ float(height)), 1.0)
    xmin = min(max(0, xmin/ float(width)), 1.0)
    ymax = max(min(1., ymax/float(height)), 0.0)
    xmax = max(min(1., xmax/float(width)), 0.0)

    return label, label_text, xmin, ymin, xmax, ymax
def extract_cuboid_bboxes(cuboids,
                          image_shape,
                          min_bbox_area=0.0002):
    labels = []
    labels_text = []
    ymins = []
    xmins = []
    ymaxs = []
    xmaxs = []
    num_cuboids = 0
    for cuboid_data in cuboids:
        label, label_text, xmin, ymin, xmax, ymax = extract_cuboid_bbox(cuboid_data, image_shape)
        if label is None:
            continue

        area = (ymax - ymin) * (xmax - xmin)
        if area < min_bbox_area:
            continue
        # only use vehicle or no vehicle        
        labels.append(1)
        labels_text.append(label_text.encode('utf8'))
        ymins.append(ymin)
        xmins.append(xmin)
        ymaxs.append(ymax)
        xmaxs.append(xmax)
        num_cuboids += 1
    return labels, labels_text, xmins, ymins, xmaxs, ymaxs, num_cuboids


def process_frame_object(frame):
    image_file = SENSOR_DATA.get_image(frame.timestamp, 3)
    if not os.path.exists(image_file):
        return None
    cuboid_data = json.loads(frame.data)

    height = 384
    width = 672
    image_shape = [height, width]
    labels, labels_text, xmins, ymins, xmaxs, ymaxs, num_cuboids = extract_cuboid_bboxes(cuboid_data,
                                                                                         image_shape)

    if num_cuboids == 0:
        print('no cuboids found for: ', frame.timestamp, frame)
        print(image_file)
        return None

    with tf.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    truncated = [0] * len(labels)
    difficult_obj = [0] * len(labels)
    poses = ['Unspecified'] * len(labels)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
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
    }))

    return example


def get_frame_objects(datetimes):
    sensor_data = SensorDataInterface('/sbox/data')
    rds = RDSService()
    rds.load(db='production')
    db = rds.session()
    frame_filter = and_(FrameObjects.data != None,
                        FrameObjects.is_cuboid,
                        FrameObjects.timestamp.in_(datetimes))
    frame_objects = db.query(FrameObjects).filter(frame_filter).all()
    db.commit()
    db.close()
    return frame_objects


def main(_):

    dataset_file = FLAGS.dataset_file

    datetimes = []
    with open(dataset_file) as f:
        for line in f:
            line = line.strip()
            digits = [int(d) for d in line.split('/')]
            dt = pytz.utc.localize(datetime(*digits))
            datetimes.append(dt)

    frame_objects = get_frame_objects(datetimes)

    frames_per_record = 500
    num_frames = 0
    record_num = int(num_frames/frames_per_record)
    try:
        import os
        os.makedirs(FLAGS.output_path)
    except:
        pass
    record_file = FLAGS.output_path + '_' + str(record_num)+ '.tf'
    writer = tf.python_io.TFRecordWriter(record_file)
    
    for i, frame in enumerate(frame_objects):
        if i % 100 == 0:
            print('Frame ', i)

        example = process_frame_object(frame)
        if example is None:
            continue
        
        if num_frames >0 and num_frames%frames_per_record == 0:
            record_num = int(num_frames/frames_per_record)
            record_file = FLAGS.output_path + '_' + str(record_num)+ '.tf'        
            print("new record: ", record_file)
            writer.close()
            writer = tf.python_io.TFRecordWriter(record_file)

        writer.write(example.SerializeToString())
        num_frames += 1

    print('Num Frames in Dataset:', num_frames)
    writer.close()


if __name__ == '__main__':
    tf.app.run()