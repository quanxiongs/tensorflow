# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import Layers as L

def build(X, n_classes,training=True ):
    keep_prob = 0.5
    
    net = L.conv(net, name = 'conv1_1',kh=3,kw=3,n_out=64)
    net = L.conv(net, name = 'conv1_2',kh=3,kw=3,n_out=64)
    net = L.pool(net, name ='pool1',kh=2,kw=2,dw=2,dh=2)
    
    net = L.conv(net, name = 'conv2_1',kh=3,kw=3,n_out=128)
    net = L.conv(net, name = 'conv2_2',kh=3,kw=3,n_out=128)
    net = L.pool(net, name ='pool2',kh=2,kw=2,dw=2,dh=2)
    
    net = L.conv(net, name = 'conv3_1',kh=3,kw=3,n_out=256)
    net = L.conv(net, name = 'conv3_2',kh=3,kw=3,n_out=256)
    net = L.pool(net, name ='pool3',kh=2,kw=2,dw=2,dh=2)
    
    net = L.conv(net, name = 'conv4_1',kh=3,kw=3,n_out=512)
    net = L.conv(net, name = 'conv4_2',kh=3,kw=3,n_out=512)
    net = L.conv(net, name = 'conv4_3',kh=3,kw=3,n_out=512)
    net = L.pool(net, name ='pool4',kh=2,kw=2,dw=2,dh=2)
    
    net = L.conv(net, name = 'conv5_1',kh=3,kw=3,n_out=512)
    net = L.conv(net, name = 'conv5_2',kh=3,kw=3,n_out=512)
    net = L.conv(net, name = 'conv5_3',kh=3,kw=3,n_out=512)
    net = L.pool(net, name ='pool5',kh=2,kw=2,dw=2,dh=2)
    
    flattened_shape = np.prod([i.value for i in net.get_shape()[1:]])
    net = tf.reshape(net, [-1,flattened_shape],name='flatten')
    
    net = L.fully_connected(net,name='FC1',n_out=4096)
    net = tf.nn.dropout(net,keep_prob= keep_prob)
    
    net = L.fully_connected(net,name='FC2',n_out=4096)
    net = tf.nn.dropout(net,keep_prob= keep_prob)
    
    net = L.fully_connected(net,name='FC3', n_out=n_classes)
    return net

    