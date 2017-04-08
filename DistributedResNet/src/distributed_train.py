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
SEED = 448

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
tf.app.flags.DEFINE_integer('num_worker_kill', 3, 'Number of workers to kill.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_integer('num_of_instances_cifar10', 50000, 'Number of data instances in Cifar10 dataset')

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

def fill_feed_dict(all_data, all_labels, image_placeholder, label_placeholder, batch_size, local_data_batch_idx, epoch_counter):
    train_batch_data, train_batch_labels, local_data_batch_idx, epoch_counter = generate_augment_train_batch(
                        all_data, all_labels, batch_size, local_data_batch_idx, epoch_counter)
    feed_dict = {image_placeholder: train_batch_data, label_placeholder: train_batch_labels}
    return epoch_counter, local_data_batch_idx, feed_dict

## Helper functions
def calc_loss(logits, labels):
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

def top_k_error(predictions, labels, k):
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

def generate_augment_train_batch(train_data, train_labels, train_batch_size, local_data_batch_idx, epoch_counter):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    '''
    offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset+train_batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
    '''
    start = local_data_batch_idx
    local_data_batch_idx += train_batch_size #next time your should start
    if local_data_batch_idx > FLAGS.num_of_instances_cifar10:
      # Finished epoch
      epoch_counter += 1
      # Shuffle the data
      perm = numpy.arange(FLAGS.num_of_instances_cifar10)
      np.random.shuffle(perm)
      train_data = train_data[perm]
      train_labels = train_labels[perm]
      # Start next epoch
      start = 0
      local_data_batch_idx = train_batch_size
      assert train_batch_size <= FLAGS.num_of_instances_cifar10
    end = local_data_batch_idx
    train_batch_tmp = train_data[start:end]
    train_batch = random_crop_and_flip(train_batch_tmp, padding_size=FLAGS.padding_size)
    batch_labels = train_labels[start:end]
    tf.logging.info("Batch shapes %s" % str(train_batch.shape))
    tf.logging.info("Standardized batch shapes %s" % str(whitening_image(train_batch).shape))
    # Most of the time return the non distorted image
    return train_batch, batch_labels, local_data_batch_idx, epoch_counter


def train(target, all_data, all_labels, cluster_spec):
    '''
    This is the main function for training
    '''
    image_placeholder = tf.placeholder(dtype=tf.float32,
                                            shape=[FLAGS.batch_size, IMG_HEIGHT,
                                                    IMG_WIDTH, IMG_DEPTH])
    label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])

    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])

    if FLAGS.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

    assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                     'num_parameter_servers'
                                                     ' must be > 0.')
    is_chief = (FLAGS.task_id == 0)
    num_examples = all_data.shape[0]

    with tf.device(
        tf.train.replica_device_setter(
        #cpu only    
#            worker_device='/job:worker/task:%d' % FLAGS.task_id,
        #with gpu enabled
            worker_device='/job:worker/task:%d/gpu:0' % FLAGS.task_id,
            cluster=cluster_spec)):

        global_step = tf.Variable(0, name="global_step", trainable=False)

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
        logits = inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)

#            vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
#            regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = calc_loss(logits, label_placeholder)

#        predictions = tf.nn.softmax(logits)
#        train_top1_error = top_k_error(predictions, label_placeholder, 1)
        
        opt = tf.train.AdamOptimizer(lr)
        if FLAGS.interval_method or FLAGS.worker_times_cdf_method:
            opt = TimeoutReplicasOptimizer(
                opt,
                global_step,
                total_num_replicas=num_workers)
        else:
#            opt = tf.train.SyncReplicasOptimizerV2(
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_replicas_to_aggregate,
                total_num_replicas=num_workers)

        # Compute gradients with respect to the loss.
        grads = opt.compute_gradients(total_loss)
        #compute weighted gradients here.
        #===============================================================================================
        '''
        #define a placeholder for weighted vector, i.e. LS solution
        weight_vec_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(num_workers,))
        grad_list = [x[0] for x in grads]
        new_grad_list = []
        #times gradient from each worker with the corresponding weight
        #which is just scalar multiplication
        for g_idx in range(len(grad_list)):
            grad_on_worker = grad_list[g_idx]
            weight = tf.slice(weight_vec_placeholder, [FLAGS.task_id], [1])
            tf.logging.info("Logging Happens Here!")
            tf.logging.info(weight[0])
            new_grad_list.append(tf.scalar_mul(weight[0], grad_on_worker))
        grad_new = []
        #regenerate the weighted gradients, merging all weighted vector
        for x_idx in range(len(grads)):
            grad_elem = grads[x_idx]
            grad_new.append((new_grad_list[x_idx], grad_elem[1]))
        '''
        #===============================================================================================
        if FLAGS.interval_method or FLAGS.worker_times_cdf_method:
            apply_gradients_op = opt.apply_gradients(grads, FLAGS.task_id, global_step=global_step, collect_cdfs=FLAGS.worker_times_cdf_method)
#            apply_gradients_op = opt.apply_gradients(grad_new, FLAGS.task_id, global_step=global_step, collect_cdfs=FLAGS.worker_times_cdf_method)
        else:
           apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
#           apply_gradients_op = opt.apply_gradients(grad_new, global_step=global_step)
        with tf.control_dependencies([apply_gradients_op]):
            train_op = tf.identity(total_loss, name='train_op')            

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        chief_queue_runners = [opt.get_chief_queue_runner()]
        init_tokens_op = opt.get_init_tokens_op()
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        test_print_op = logging_ops.Print(0, [0], message="Test print success")
        if is_chief:
            local_init_op = opt.chief_init_op
        else:
            local_init_op = opt.local_step_init_op

        local_init_opt = [local_init_op]
        ready_for_local_init_op = opt.ready_for_local_init_op

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 local_init_op=local_init_op,
                                 ready_for_local_init_op=ready_for_local_init_op,
                                 logdir=FLAGS.train_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 global_step=global_step,
                                 saver=saver,
                                 save_model_secs=FLAGS.save_interval_secs)
        tf.logging.info('%s Supervisor' % datetime.now())
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)
        sess = sv.prepare_or_wait_for_session(target, config=sess_config)
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        sv.start_queue_runners(sess, queue_runners)
        tf.logging.info('Started %d queues for processing input data.',
                        len(queue_runners))

        if is_chief:
            if not FLAGS.interval_method or FLAGS.worker_times_cdf_method:
                sv.start_queue_runners(sess, chief_queue_runners)
            sess.run(init_tokens_op)

        timeout_client, timeout_server = launch_manager(sess, FLAGS)
        next_summary_time = time.time() + FLAGS.save_summaries_secs
        begin_time = time.time()
        cur_iteration = -1
        local_data_batch_idx = 0
        epoch_counter = 0
        iterations_finished = set()

        if FLAGS.task_id == 0 and FLAGS.interval_method:
            opt.start_interval_updates(sess, timeout_client)   
        '''
        np.random.seed(SEED)
        b = np.ones(int(num_batches_per_epoch))
        interval = np.arange(0, int(num_batches_per_epoch))
        idx_list = np.random.choice(interval, int(num_workers), replace=False)     
        '''
        while not sv.should_stop():
        #    try:
            sys.stdout.flush()
            tf.logging.info("A new iteration...")
            cur_iteration += 1

            if FLAGS.worker_times_cdf_method:
                sess.run([opt._wait_op])
                timeout_client.broadcast_worker_dequeued_token(cur_iteration)
            start_time = time.time()
            epoch_counter, local_data_batch_idx, feed_dict = fill_feed_dict(
                all_data, all_labels, image_placeholder, label_placeholder, FLAGS.batch_size, local_data_batch_idx, epoch_counter)

            run_options = tf.RunOptions()
            run_metadata = tf.RunMetadata()
            #=============================================================================================== 
            '''
            LS_start_time = time.time()
            interval_2 = np.arange(0, int(num_workers))
            workers_to_kill = np.random.choice(interval_2, FLAGS.num_worker_kill, replace=False)
            #interval_2 = np.arange(0, WORKER_NUM)
            #workers_to_kill = np.random.choice(interval_2, NUM_WORKER_KILL, replace=False)
            A = np.zeros((int(num_workers), int(num_batches_per_epoch)))
            for i in range(A.shape[0]):
              if i == A.shape[0]-1:
                A[i][idx_list[i]] = 1
                A[i][idx_list[0]] = 1
              else:
                A[i][idx_list[i]] = 1
                A[i][idx_list[i+1]] = 1

            for i in range(len(idx_list)):
              element = idx_list[i]
              if element == A.shape[1]-1:
                idx_list[i] = 0
              else:
                idx_list[i] += 1

            for k in workers_to_kill:
              A[k] = 0

            A_for_calc = np.transpose(A)
            ls_solution = np.dot(np.linalg.pinv(A_for_calc), b)
            tf.logging.info("workers killed this iteration:")
            tf.logging.info(str(workers_to_kill))
            tf.logging.info("The matrix to solve:")
            for item in A_for_calc:
              tf.logging.info(str(item))
            tf.logging.info("Solution of LS:")
            tf.logging.info(str(ls_solution)) 
            LS_duration = time.time() - LS_start_time
            tf.logging.info("LS run time: %s" % str(LS_duration))
            '''
            #===============================================================================================             

            if FLAGS.timeline_logging:
                run_options.trace_level=tf.RunOptions.FULL_TRACE
                run_options.output_partition_graphs=True

            #feed_dict[weight_vec_placeholder] = ls_solution
            tf.logging.info("RUNNING SESSION... %f" % time.time())
            tf.logging.info("Data batch index: %s, Current epoch idex: %s" % (str(epoch_counter), str(local_data_batch_idx)))
            loss_value, step = sess.run(
                #[train_op, global_step], feed_dict={feed_dict, x}, run_metadata=run_metadata, options=run_options)
                [train_op, global_step], feed_dict=feed_dict, run_metadata=run_metadata, options=run_options)
            tf.logging.info("DONE RUNNING SESSION...")

            if FLAGS.worker_times_cdf_method:
                timeout_client.broadcast_worker_finished_computing_gradients(cur_iteration)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            finish_time = time.time()
            if FLAGS.timeline_logging:
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('%s/worker=%d_timeline_iter=%d.json' % (FLAGS.train_dir, FLAGS.task_id, step), 'w'):
                    f.write(ctf)
            if step > FLAGS.max_steps:
                break

            duration = time.time() - start_time
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('Worker %d: %s: step %d, loss = %f'
                            '(%.1f examples/sec; %.3f  sec/batch)')
            tf.logging.info(format_str %
                        (FLAGS.task_id, datetime.now(), step, loss_value,
                            examples_per_sec, duration))
            if is_chief and next_summary_time < time.time() and FLAGS.should_summarize:
                tf.logging.info('Running Summary operation on the chief.')
                summary_str = sess.run(summary_op)
                sv.summary_computed(sess, summary_str)
                tf.logging.info('Finished running Summary operation.')
                next_summary_time += FLAGS.save_summaries_secs
        #    except tf.errors.DeadlineExceededError:
        #        tf.logging.info("Killed at time %f" % time.time())
                #sess.reset_kill()
        #    except:
        #        tf.logging.info("Unexpected error: %s" % str(sys.exc_info()[0]))
                #sess.reset_kill()
        if is_chief:
            tf.logging.info('Elapsed Time: %f' % (time.time()-begin_time))
        sv.stop()

        if is_chief:
            saver.save(sess,
                        os.path.join(FLAGS.train_dir, 'model.ckpt'),
                        global_step=global_step)



