# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
#Revised by: Hongyi Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd

from threading import Timer
from sync_replicas_optimizer_modified.sync_replicas_optimizer_modified import TimeoutReplicasOptimizer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from timeout_manager import launch_manager

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('worker_times_cdf_method', False, 'Track worker times cdf')
tf.app.flags.DEFINE_boolean('interval_method', False, 'Use the interval method')
tf.app.flags.DEFINE_boolean('should_summarize', False, 'Whether Chief should write summaries.')
tf.app.flags.DEFINE_boolean('timeline_logging', False, 'Whether to log timeline of events.')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('rpc_port', 1235,
                           """Port for timeout communication""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 20,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 300,
                            'Save summaries interval seconds.')

#=========================================================================#
# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
#tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
#                          'Initial learning rate.')
# For flowers
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.999,
                          'Learning rate decay factor.')
#=========================================================================#

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    '''
    def build_train_validation_graph(self, num_examples):
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)
    '''
    

    def train(self, target, all_data, all_labels, cluster_spec):
        '''
        This is the main function for training
        '''
        num_workers = len(cluster_spec.as_dict()['worker'])
        num_parameter_servers = len(cluster_spec.as_dict()['ps'])

        if FLAGS.num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate\

        assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')
        is_chief = (FLAGS.task_id == 0)
        num_examples = all_data.shape[0]

        with tf.device(
          tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % FLAGS.task_id,
            cluster=cluster_spec)):

            global_step = tf.Variable(0, trainable=False)

            num_batches_per_epoch = (num_examples / FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_replicas_to_aggregate)
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
            # Logits of training data and valiation data come from the same graph. The inference of
            # validation data share all the weights with train data. This is implemented by passing
            # reuse=True to the variable scopes of train graph
            logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
#            vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

            # The following codes calculate the train loss, which is consist of the
            # softmax cross entropy and the relularization loss
#            regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = self.loss(logits, self.label_placeholder)
            self.full_loss = loss
 #           self.full_loss = tf.add_n([loss] + regu_losses)

            predictions = tf.nn.softmax(logits)
            self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

            ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
            train_ema_op = ema.apply([total_loss, top1_error])

            opt = tf.train.AdamOptimizer(lr)
            if FLAGS.interval_method or FLAGS.worker_times_cdf_method:
                opt = TimeoutReplicasOptimizer(
                    opt,
                    global_step,
                    total_num_replicas=num_workers)
            else:
                opt = tf.train.SyncReplicasOptimizerV2(
                    opt,
                    replicas_to_aggregate=num_replicas_to_aggregate,
                    total_num_replicas=num_workers)

            # Compute gradients with respect to the loss.
            grads = opt.compute_gradients(self.full_loss)
            if FLAGS.interval_method or FLAGS.worker_times_cdf_method:
                apply_gradients_op = opt.apply_gradients(grads, FLAGS.task_id, global_step=global_step, collect_cdfs=FLAGS.worker_times_cdf_method)
            else:
                apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')            
            # Validation loss
            '''
            self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
            vali_predictions = tf.nn.softmax(vali_logits)
            self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
            '''

#            self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
#                                                                    self.train_top1_error, lr)

            # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
            # summarizing operations by running summary_op. Initialize a new session
            saver = tf.train.Saver(tf.global_variables())
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            sess = tf.Session()

            # If you want to load from a checkpoint
            if FLAGS.is_use_ckpt is True:
                saver.restore(sess, FLAGS.ckpt_path)
                print 'Restored from checkpoint...'
            else:
                sess.run(init)

            # This summary writer object helps write summaries on tensorboard
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


            # These lists are used to save a csv file at last
            step_list = []
            train_error_list = []
            val_error_list = []

            print 'Start training...'
            print '----------------------------'

            for step in xrange(FLAGS.train_steps):

                train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                            FLAGS.train_batch_size)


                validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                               vali_labels, FLAGS.validation_batch_size)

                # Want to validate once before training. You may check the theoretical validation
                # loss first
                '''
                if step % FLAGS.report_freq == 0:

                    if FLAGS.is_full_validation is True:
                        validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                                top1_error=self.vali_top1_error, vali_data=vali_data,
                                                vali_labels=vali_labels, session=sess,
                                                batch_data=train_batch_data, batch_label=train_batch_labels)

                        vali_summ = tf.Summary()
                        vali_summ.value.add(tag='full_validation_error',
                                            simple_value=validation_error_value.astype(np.float))
                        summary_writer.add_summary(vali_summ, step)
                        summary_writer.flush()

                    else:
                        _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                         self.vali_top1_error,
                                                                     self.vali_loss],
                                                    {self.image_placeholder: train_batch_data,
                                                     self.label_placeholder: train_batch_labels,
                                                     self.vali_image_placeholder: validation_batch_data,
                                                     self.vali_label_placeholder: validation_batch_labels,
                                                     self.lr_placeholder: FLAGS.init_lr})

                    val_error_list.append(validation_error_value)
                '''

                start_time = time.time()

                _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                               self.full_loss, self.train_top1_error],
                                    {self.image_placeholder: train_batch_data,
                                      self.label_placeholder: train_batch_labels,
                                      self.vali_image_placeholder: validation_batch_data,
                                      self.vali_label_placeholder: validation_batch_labels,
                                      self.lr_placeholder: FLAGS.init_lr})
                duration = time.time() - start_time


                if step % FLAGS.report_freq == 0:
                    summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                        self.label_placeholder: train_batch_labels,
                                                        self.vali_image_placeholder: validation_batch_data,
                                                        self.vali_label_placeholder: validation_batch_labels,
                                                        self.lr_placeholder: FLAGS.init_lr})
                    summary_writer.add_summary(summary_str, step)

                    num_examples_per_step = FLAGS.train_batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                    print format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                        sec_per_batch)
                    print 'Train top1 error = ', train_error_value
                    '''
                    print 'Validation top1 error = %.4f' % validation_error_value
                    print 'Validation loss = ', validation_loss_value
                    print '----------------------------'
                    '''

                    step_list.append(step)
                    train_error_list.append(train_error_value)



                if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                    FLAGS.init_lr = 0.1 * FLAGS.init_lr
                    print 'Learning rate decayed to ', FLAGS.init_lr
                '''
                # Save checkpoints every 10000 steps
                if step % 10000 == 0 or (step + 1) == FLAGS.train_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                    df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                    'validation_error': val_error_list})
                    df.to_csv(train_dir + FLAGS.version + '_error.csv')
                '''


    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print '%i test batches in total...' %num_batches

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print 'Model restored from ', FLAGS.test_ckpt_path

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print '%i batches finished!' %step
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
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


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(10000 - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label

    '''
    def train_operation(self, global_step, total_loss, top1_error, lr):
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.AdamOptimizer(lr)

        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op
    '''

    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)

'''
maybe_download_and_extract()

# Initialize the Train object
train = Train()
# Start the training session
train.train()
'''




