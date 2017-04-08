# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.python.platform
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def conv(X,name,kw,kh,n_out,dw=1,dh=1,activation_fn = tf.nn.relu):
    n_in = X.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',[kh,kw,n_in,n_out],tf.float32, initializer=xavier_initializer())
        biase = tf.get_variable('bias',[n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(X,weights,(1,dh,dw,1),padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv,biases))
        return activation

def pool(X,name,kw,kh,dh,dw):
    return tf.nn.max_pool(X,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='VALID',name=name)

def fully_connected(X,name, n_out, activation_fn = tf.nn.relu):
    n_in = X.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in,n_out],tf.float32, xavier_initializer())
        biases = tf.get_variable('biases',[n_out],tf.float32,tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(X,weights),biases)
        return activation_fn(logits)

def loss(X,y):
    softmax = tf.nn.softmax_cross_entropy_with_logits(X,y,name='softmax')
    loss = tf.reduce_mean(softmax, name='loss')
    return loss

def topK_error(yhat, y,K=5):
    score = tf.cast(tf.nn.in_top_k(yhat,y,K),tf.float32)
    accuracy = tf.reduce_mean(score)
    error = 1.0 -accuracy
    return error