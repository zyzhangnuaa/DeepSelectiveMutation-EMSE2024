from keras.layers import Input
import importlib
from keras.datasets import mnist, cifar10
from keras import backend as K
from keras.utils import to_categorical
import scipy.io as sio
import numpy as np
import os
from keras_preprocessing import image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


def preprocess_image2(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 32, 32, 3)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

# 加载新生成的测试输入
def load_new_data(data_path, dataset):
    x_test = []
    y_test = []
    img_dir = data_path
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)
    for i in range(img_num):
        path = os.path.join(img_dir, img_paths[i])
        img_paths2 = os.listdir(path)
        img_num2 = len(img_paths2)
        for j in range(img_num2):
            img_path = os.path.join(path, img_paths2[j])
            print('aaaadasf:', img_path)
            img_name = img_path.split('\\')[-1].split('.')[0]
            print('aaaadasfname:', img_name)
            mannual_label = int(img_name.split('_')[1])
            print(img_path, img_name, mannual_label)
            if dataset == 'mnist':
                img = preprocess_image(img_path)
            else:
                img = preprocess_image2(img_path)
            x_test.append(img[0])
            y_test.append(mannual_label)
    x_test_new = np.asarray(x_test)
    y_test_new = np.asarray(y_test)
    print('x_test_new_shape', x_test_new.shape)
    print('y_test_new_shape', y_test_new.shape)

    return x_test_new, y_test_new


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


# def load_model(model_path, subject_name, dataset):
#     if dataset == 'mnist':
#         img_rows, img_cols = 28, 28
#         input_shape = (img_rows, img_cols, 1)
#     elif dataset == 'cifar10' or dataset == 'svhn':
#         img_rows, img_cols = 32, 32
#         input_shape = (img_rows, img_cols, 3)
#     input_tensor = Input(shape=input_shape)
#     m1 = importlib.import_module('experiment_model.' + subject_name)
#     _, model = m1.main(model_path, input_tensor=input_tensor)
#     return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # change output to one-hot vector.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
        x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
        input_shape = (3, 32, 32)
    else:
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
        input_shape = (32, 32, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # change output to one-hot vector.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test
