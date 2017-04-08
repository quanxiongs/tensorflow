# -*- coding: utf-8 -*-


import _pickle as cpickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    with open(filename,'rb') as f: #should we decode binary file?
        datadict = cpickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("float")
        #transpose get the dimension in the order of index
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,2):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs) 
  Ytr = np.concatenate(ys)
  
  """
  this allows collapse the first dimension
  for example dim [3,3,2] -> [9,2]
  so maybe we don't need this right now
  """
  
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=3000, num_validation=1000, num_test=1000):
    cifar10_dir = 'C:/SparkCourse/Tensorflow/My_vggnet/cifar10'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



#data = get_CIFAR10_data() 
#data['X_train'].shape
#data['y_train'].shape
#data['X_val'].shape
#data['y_val'].shape
#data['X_test'].shape
#data['y_test'].shape




#should read sequential data
#loading error of pickle library


#itertools as it
#https://docs.python.org/2/library/itertools.html

def batchmaker(batch_size):
    mask = np.random.randint(low=0,high=3000,size=batch_size)
    data = get_CIFAR10_data()
    data[]



