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
"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""

import logging
import tensorflow as tf
import numpy as np
from object_detection import eval_util
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.utils import object_detection_evaluation
from object_detection.utils import visualization_utils as vis_utils
slim = tf.contrib.slim
# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'open_images_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator
}


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                categories,
                                ignore_groundtruth=False):
    """Restores the model in a tensorflow session.

    Args:
      model: model to perform predictions with.
      create_input_dict_fn: function to create input tensor dictionaries.
      ignore_groundtruth: whether groundtruth should be ignored.

    Returns:
      tensor_dict: A tensor dictionary with evaluations.
    """
    input_dict = create_input_dict_fn()
    prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
    input_dict = prefetch_queue.dequeue()
    original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
    preprocessed_image = model.preprocess(tf.to_float(original_image))
    prediction_dict = model.predict(preprocessed_image)
    detections = model.postprocess(prediction_dict)

    groundtruth = None
    if not ignore_groundtruth:
        groundtruth = {
            fields.InputDataFields.groundtruth_boxes:
                input_dict[fields.InputDataFields.groundtruth_boxes],
            fields.InputDataFields.groundtruth_classes:
                input_dict[fields.InputDataFields.groundtruth_classes],
            fields.InputDataFields.groundtruth_area:
                input_dict[fields.InputDataFields.groundtruth_area],
            fields.InputDataFields.groundtruth_is_crowd:
                input_dict[fields.InputDataFields.groundtruth_is_crowd],
            fields.InputDataFields.groundtruth_difficult:
                input_dict[fields.InputDataFields.groundtruth_difficult]
        }
        if fields.InputDataFields.groundtruth_group_of in input_dict:
            groundtruth[fields.InputDataFields.groundtruth_group_of] = (
                input_dict[fields.InputDataFields.groundtruth_group_of])
        if fields.DetectionResultFields.detection_masks in detections:
            groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
                input_dict[fields.InputDataFields.groundtruth_instance_masks])

    images = tf.unstack(preprocessed_image)
    groundtruth_boxes = [groundtruth['groundtruth_boxes']]
    groundtruth_classes = [groundtruth['groundtruth_classes']]
    groundtruth_scores = [tf.ones_like(c, dtype=tf.float32) for c in groundtruth_classes]
    vis_utils.draw_bounding_boxes_on_image_tensors("groundtruths",
                                                   images,
                                                   groundtruth_boxes,
                                                   groundtruth_classes,
                                                   groundtruth_scores,
                                                   categories)
    # images = tf.unstack(preprocessed_image)
    detection_boxes = tf.unstack(detections['detection_boxes'])
    detection_classes = tf.unstack(tf.cast(detections['detection_classes'] + 1, tf.int64))
    detection_scores = tf.unstack(detections['detection_scores'])
    vis_utils.draw_bounding_boxes_on_image_tensors("predictions",
                                                   images,
                                                   detection_boxes,
                                                   detection_classes,
                                                   detection_scores,
                                                   categories,
                                                   min_score_thresh=0.5)

    return eval_util.result_dict_for_single_example(
        original_image,
        input_dict[fields.InputDataFields.source_id],
        detections,
        groundtruth,
        class_agnostic=(
            fields.DetectionResultFields.detection_classes not in detections),
        scale_to_absolute=True)


def get_evaluators(eval_config, categories):
    """Returns the evaluator class according to eval_config, valid for categories.

    Args:
      eval_config: evaluation configurations.
      categories: a list of categories to evaluate.
    Returns:
      An list of instances of DetectionEvaluator.

    Raises:
      ValueError: if metric is not in the metric class dictionary.
    """
    eval_metric_fn_key = eval_config.metrics_set
    if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
        raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    return [
        EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](
            categories=categories)
    ]


def evaluate(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir):
    """Evaluation function for detection models.

    Args:
      create_input_dict_fn: a function to create a tensor input dictionary.
      create_model_fn: a function that creates a DetectionModel.
      eval_config: a eval_pb2.EvalConfig protobuf.
      categories: a list of category dictionaries. Each dict in the list should
                  have an integer 'id' field and string 'name' field.
      checkpoint_dir: directory to load the checkpoints to evaluate from.
      eval_dir: directory to write evaluation metrics summary to.

    Returns:
      metrics: A dictionary containing metric names and values from the latest
        run.
    """

    model = create_model_fn()

    if eval_config.ignore_groundtruth and not eval_config.export_path:
        logging.fatal('If ignore_groundtruth=True then an export_path is '
                      'required. Aborting!!!')

    tensor_dict = _extract_prediction_tensors(
        model=model,
        create_input_dict_fn=create_input_dict_fn,
        categories=categories,
        ignore_groundtruth=eval_config.ignore_groundtruth)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    def _process_batch(tensor_dict, sess, batch_index, counters, number_of_evaluations):
        """Evaluates tensors in tensor_dict, visualizing the first K examples.

        This function calls sess.run on tensor_dict, evaluating the original_image
        tensor only on the first K examples and visualizing detections overlaid
        on this original_image.

        Args:
          tensor_dict: a dictionary of tensors
          sess: tensorflow session
          batch_index: the index of the batch amongst all batches in the run.
          counters: a dictionary holding 'success' and 'skipped' fields which can
            be updated to keep track of number of successful and failed runs,
            respectively.  If these fields are not updated, then the success/skipped
            counter values shown at the end of evaluation will be incorrect.

        Returns:
          result_dict: a dictionary of numpy arrays
        """
        tensor_dict = dict(tensor_dict)
        if batch_index < eval_config.num_visualizations:
            tensor_dict['summary'] = summary_op
        try:
            result_dict = sess.run(tensor_dict)
            counters['success'] += 1
        except tf.errors.InvalidArgumentError:
            logging.info('Skipping image')
            counters['skipped'] += 1
            return {}

        if 'summary' in result_dict:
            print 'Printing Summary: ', number_of_evaluations
            summary = result_dict['summary']
            summary_writer = tf.summary.FileWriter(eval_dir)
            summary_writer.add_summary(summary, number_of_evaluations)
            summary_writer.close()

        return result_dict

    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)
    if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    def _restore_latest_checkpoint(sess):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)

    metrics = eval_util.repeated_checkpoint_run(
        tensor_dict=tensor_dict,
        summary_dir=eval_dir,
        evaluators=get_evaluators(eval_config, categories),
        batch_processor=_process_batch,
        checkpoint_dirs=[checkpoint_dir],
        variables_to_restore=None,
        restore_fn=_restore_latest_checkpoint,
        num_batches=eval_config.num_examples,
        eval_interval_secs=eval_config.eval_interval_secs,
        max_number_of_evaluations=(1 if eval_config.ignore_groundtruth else
                                   eval_config.max_evals
                                   if eval_config.max_evals else None),
        master=eval_config.eval_master,
        save_graph=eval_config.save_graph,
        save_graph_dir=(eval_dir if eval_config.save_graph else ''))

    return metrics
