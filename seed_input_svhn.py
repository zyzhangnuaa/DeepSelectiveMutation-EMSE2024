import importlib
import os
import random
from imageio import imsave
import keras
import h5py
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K
from tensorflow.python.keras import Input
import scipy.io as sio
from tensorflow.keras.models import load_model


def load_svhn(one_hot=False):
    train = sio.loadmat('Datasets/svhn/train_32x32.mat')
    test = sio.loadmat('Datasets/svhn/test_32x32.mat')

    train_data = train['X']
    train_label = train['y']
    test_data = test['X']
    test_label = test['y']

    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 2)

    test_data = test_data / 255.
    train_data = train_data / 255.

    for i in range(train_label.shape[0]):
        if train_label[i][0] == 10:
            train_label[i][0] = 0

    for i in range(test_label.shape[0]):
        if test_label[i][0] == 10:
            test_label[i][0] = 0

    num_labels = 10

    if one_hot:
        train_label = (np.arange(num_labels) == train_label[:, ]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:, ]).astype(np.float32)

    return train_data, train_label, test_data, test_label


def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape((32, 32, 3))  # original shape (1,img_rows, img_cols,1)


def input_reshape_test(x_test, y_test, num_classes):
    img_rows, img_cols = 32, 32
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_test, y_test


def get_svhn_non_uniform_ts(seed_input_path, c, each_num):
    x_train, y_train, x_test, y_test = load_svhn(one_hot=False)
    y_test = y_test.flatten()

    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    subject_name = 'svhn'
    original_model_path = os.path.join('original_model', subject_name + '_original.h5')
    model = load_model(original_model_path)

    # 根据标签类别对测试数据进行排序
    y_sort_index = np.argsort(y_test)
    y_test_sort = y_test[y_sort_index]
    x_test_sort = x_test[y_sort_index]

    ori_predict = model.predict(x_test_sort).argmax(axis=-1)
    correct_index = np.where(ori_predict == y_test_sort)[0]
    print("csdgvuydf:", len(correct_index))

    xx_test, yy_test = input_reshape_test(x_test_sort, y_test_sort, 10)
    predicted = np.asarray(model.predict(xx_test))
    predicted = np.sort(predicted, axis=1)

    print('sddsvdf:', predicted)

    # x_test_sort = x_test_sort.reshape(x_test_sort.shape[0], img_rows, img_cols, 3)
    # x_test_sort = x_test_sort.astype('float32')
    # x_test_sort /= 255

    # 每个类别测试数据的数量
    num = [0 for index in range(10)]
    for i in range(len(y_test_sort)):
        num[y_test_sort[i]] += 1
    print(num)

    # 每个类别的数据的起始位置
    begin_index = 0
    if c != 0:
        for i in range(c):
            begin_index += num[i]
    print(begin_index)

    pre_list = {}
    for index in range(begin_index, begin_index + num[c]):
        if index in correct_index:
            pre_list[index] = predicted[index][-1]/predicted[index][-2]

    pre_list = dict(sorted(pre_list.items(), key=lambda item:item[1]))
    print('dhucvygdv:', pre_list)

    # 选取的图片的idex
    pre_list_seed_idex = dict(list(pre_list.items())[:each_num]).keys()
    print('hsd:', pre_list_seed_idex)

    # 保存图片
    save_path = os.path.join(seed_input_path, str(c))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in pre_list_seed_idex:
        img_deprocessed = deprocess_image(x_test_sort[i])
        imsave(save_path + '/' + str(i) + '_' + str(y_test_sort[i]) + '.png', img_deprocessed)


if __name__ == '__main__':
    seed_num = 100
    each_num = int(seed_num / 10)
    seed_input_path = os.path.join('seed_input', 'svhn', 'seeds_' + str(seed_num))
    original_model_path = os.path.join('original_model', 'svhn')
    for i in range(10):
        get_svhn_non_uniform_ts(seed_input_path, i, each_num)

