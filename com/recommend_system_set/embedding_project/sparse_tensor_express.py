#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Mike
# @Contact : yinhuaxi@geotmt.com
# @Time    : 2020/1/7 14:33
# @File    : sparse_tensor_express.py
# tf 1.x
import tensorflow as tf
import pandas as pd

csv = [
    "1,harden|james|curry",
    "2,wrestbrook|harden|durant",
    "3,|paul|towns",
]

TAG_SET = ["harden", "james", "curry", "durant", "paul", "towns", "wrestbrook"]
csv = tf.constant(csv)


def sparse_from_csv(csv):
    ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=TAG_SET, default_value=-1
    )  # 构造查找表
    split_tags = tf.string_split(post_tags_str, tf.constant("|"))
    return tf.SparseTensor(indices=split_tags.indices,
                           values=table.lookup(split_tags.values),  # 这里给出了不同值通过表查到的index
                           dense_shape=split_tags.dense_shape)


TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET)], TAG_EMBEDDING_DIM))

# 得到embedding值
tags = sparse_from_csv(csv)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)

with tf.Session() as s:
    s.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(s.run([embedded_tags]))
