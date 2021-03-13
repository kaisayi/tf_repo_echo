#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Obito
Create Time: 2021/3/13 16:14
"""
import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", "cifar_10", "name of model")
flags.DEFINE_integer("num_epochs", 20, "number of training epochs")
flags.DEFINE_bool("log_device_placement", False, "whether log device info")

def create_model(is_training, features, num_labels=10):
    return 0.0, 0.0, 0.0


def model_fn_builder(train_params, optimizer_params):
    """
    :return: 定义模型
    """

    def _model_fn(features, labels, mode, params):
        init_checkpoint = train_params.get("init_checkpoint")
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("\tname = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        logging.info("******** IS TRAINING: %d *********" % is_training)

        (total_loss, per_loss, probabilities) = create_model(is_training, features)
        if mode == tf.estimator.ModeKeys.TRAIN:
            pass

    return _model_fn

def input_fn_builder():
    """
    :return: 定义输入结构
    """

    def _input_fn():
        pass

    return _input_fn


def main(_):

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=FLAGS.num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.job_dir,
        session_config=sess_config)


    pass

if __name__ == '__main__':
    # 指定必须传递的参数
    flags.mark_flag_as_required("model_name")

    app.run(main)



