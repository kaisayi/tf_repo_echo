#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author: Obito
Create Time: 2021/3/13 16:43
"""

import os
import tensorflow as tf

class BaseDataSet(object):

    def __init__(self, file_pattern, mode, **kwargs):
        self.file_pattern = file_pattern
        self.mode = mode

    def get_filenames(self):
        return tf.io.matching_files(self.file_pattern)

    def parser(self, serialized_example):
        raise NotImplementedError

    def make_batch(self, batch_size):
        filenames = self.get_filenames()
        dataset = tf.data.TFRecordDataset(filenames).repeat()

        # parse records
        dataset = dataset.map(
            self.parser,
            num_parallel_calls=batch_size
        )

        # potentially shuffle
        if self.mode == "train":
            min_queue_examples = int()



