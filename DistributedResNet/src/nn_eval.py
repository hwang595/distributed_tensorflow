# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library to evaluate nn on mnist validation data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import time

import sys
import numpy as np
import tensorflow as tf
from distributed_train import *
from resnet import *


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

start_time = time.time()

def fill_feed_dict(eval_data, eval_labels, images_pl, labels_pl, batch_size):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = generate_vali_batch(eval_data, eval_labels, batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed
  }
  return feed_dict

def do_eval(saver,
            writer,
            val_acc,
            val_loss,
            images_placeholder,
            labels_placeholder,
            eval_data,
            eval_labels,
            prev_global_step=-1):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  try:
    with tf.Session() as sess:

      # Load checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                           ckpt.model_checkpoint_path))
      else:
        print('No checkpoint file found')
        sys.stdout.flush()
        return -1

      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      # Don't evaluate on the same checkpoint
      if prev_global_step == global_step:
        return prev_global_step

      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
      sys.stdout.flush()

      # Compute accuracy
      num_examples = eval_data.shape[0]
      feed_dict = fill_feed_dict(eval_data, eval_labels, 
                                       images_placeholder,
                                       labels_placeholder,
                                       num_examples)
      acc, loss = sess.run([val_acc, val_loss], feed_dict=feed_dict)

      print('Num examples: %d  Precision @ 1: %f Loss: %f Time: %f' %
            (num_examples, acc, loss, time.time() - start_time))
      sys.stdout.flush()

      # Summarize accuracy
      summary = tf.Summary()
      summary.value.add(tag="Validation Accuracy", simple_value=float(acc))
      summary.value.add(tag="Validation Loss", simple_value=float(loss))
      writer.add_summary(summary, global_step)
    return global_step

  except Exception as e:
    print(e.__doc__)
    print(e.message)

def generate_vali_batch(vali_data, vali_label, vali_batch_size):
  '''
  If you want to use a random batch of validation data to validate instead of using the
  whole validation data, this function helps you generate that batch
  :param vali_data: 4D numpy array
  :param vali_label: 1D numpy array
  :param vali_batch_size: int
  :return: 4D numpy array and 1D numpy array
  '''
#  offset = np.random.choice(10000 - vali_batch_size, 1)[0]
  offset = np.random.choice(10001 - vali_batch_size, 1)[0]
  vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
  vali_label_batch = vali_label[offset:offset+vali_batch_size]
  return vali_data_batch, vali_label_batch

def loss(logits, labels):
  '''
  Calculate the cross entropy loss given logits and true labels
  :param logits: 2D tensor with shape [batch_size, num_labels]
  :param labels: 1D tensor with shape [batch_size]
  :return: loss tensor with shape [1]
  '''
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  return cross_entropy_mean

def evaluate(eval_data, eval_labels):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Graph creation
    num_examples = eval_data.shape[0]
    batch_size = num_examples
    images_placeholder, labels_placeholder = generate_vali_batch(
      eval_data, eval_labels, batch_size)
    logits = inference(images_placeholder, FLAGS.num_residual_blocks, reuse=False)
#    validation_loss = mnist.loss(logits, labels_placeholder)
    predictions = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(predictions, labels_placeholder, 1)
    tmp_c = tf.reduce_sum(tf.cast(correct, tf.int32))
    validation_accuracy = tf.reduce_sum(tmp_c) / tf.constant(batch_size)
    validation_loss = loss(logits, labels_placeholder)

    # Reference to sess and saver
    sess = tf.Session()
    saver = tf.train.Saver()

    # Create summary writer
    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           graph_def=graph_def)
    step = -1
    while True:
      step = do_eval(
        saver, summary_writer, validation_accuracy, validation_loss, images_placeholder, 
        labels_placeholder, eval_data, eval_labels, prev_global_step=step)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
