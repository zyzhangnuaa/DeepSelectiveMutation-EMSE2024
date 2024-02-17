import glob
import importlib
import os

import h5py
import numpy as np
from keras.datasets import mnist, cifar10
from tensorflow.python.keras import Input
from keras.utils import to_categorical
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)
m1 = importlib.import_module('experiment_model.mnist')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

_, model = m1.main('original_model/mnist_original.h5', input_tensor=input_tensor)
model.summary()


