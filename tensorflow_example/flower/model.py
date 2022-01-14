#coding:utf-8
 
from tensorflow.contrib.layers.python.layers import batch_norm  ###0706(tf==2.0)

# from tensorflow.python.ops.layers.python.layers import batch_norm

import tensorflow as tf
import inspect
import os
import numpy as np
import time
 
 
def model4(x, N_CLASSES, is_trian = False):
    x = tf.contrib.layers.conv2d(x, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
    x = batch_norm(x, decay=0.9, updates_collections=None, is_training=is_trian)  # 训练阶段is_trainging设置为true,训练完毕后使用模型时设置为false
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')
 
    x1_1 = tf.contrib.layers.conv2d(x, 64, [1, 1], 1, 'SAME', activation_fn=tf.nn.relu)  # 1X1 核
    x1_1 = batch_norm(x1_1, decay=0.9, updates_collections=None, is_training=is_trian)
    x3_3 = tf.contrib.layers.conv2d(x, 64, [3, 3], 1, 'SAME', activation_fn=tf.nn.relu)  # 3x3 核
    x3_3 = batch_norm(x3_3, decay=0.9, updates_collections=None, is_training=is_trian)
    x5_5 = tf.contrib.layers.conv2d(x, 64, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)  # 5x5 核
    x5_5 = batch_norm(x5_5, decay=0.9, updates_collections=None, is_training=is_trian)
    x = tf.concat([x1_1, x3_3, x5_5], axis=-1)  # 连接在一起，得到64*3=192个通道
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')
 
    x1_1 = tf.contrib.layers.conv2d(x, 128, [1, 1], 1, 'SAME', activation_fn=tf.nn.relu)
    x1_1 = batch_norm(x1_1, decay=0.9, updates_collections=None, is_training=is_trian)
    x3_3 = tf.contrib.layers.conv2d(x, 128, [3, 3], 1, 'SAME', activation_fn=tf.nn.relu)
    x3_3 = batch_norm(x3_3, decay=0.9, updates_collections=None, is_training=is_trian)
    x5_5 = tf.contrib.layers.conv2d(x, 128, [5, 5], 1, 'SAME', activation_fn=tf.nn.relu)
    x5_5 = batch_norm(x5_5, decay=0.9, updates_collections=None, is_training=is_trian)
    x = tf.concat([x1_1, x3_3, x5_5], axis=-1)
    x = tf.contrib.layers.max_pool2d(x, [2, 2], stride=2, padding='SAME')
 
    shp = x.get_shape()
    x = tf.reshape(x, [-1, shp[1]*shp[2]*shp[3]])  # flatten
    x = tf.contrib.layers.fully_connected(x, N_CLASSES, activation_fn=None)  # output logist without softmax
    return x
 
 
def model2(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
    
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
     
        
    # full connect
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return logits
 