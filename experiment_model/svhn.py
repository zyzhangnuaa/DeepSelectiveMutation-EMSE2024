import os
import keras
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np
import scipy.io as sio

image_size = 32
num_labels = 10


def load_svhn(one_hot=False):
    train = sio.loadmat('../Datasets/svhn/train_32x32.mat')
    test = sio.loadmat('../Datasets/svhn/test_32x32.mat')

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


def main(model_local):
    print('aaaaaaaaaaaaa', model_local)
    nb_classes = 10
    # convolution kernel size
    kernel_size = (3, 3)

    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, shuffled and split between train and test sets
    x_train, y_train, x_test, y_test = load_svhn(one_hot=True)

    input_shape = (img_rows, img_cols, 3)

    if not os.path.exists(model_local):
        batch_size = 128
        nb_epoch = 12
        model = Sequential()
        model.add(Convolution2D(64, kernel_size, activation='relu', name='block1_conv1', input_shape=input_shape))
        model.add(Convolution2D(64, kernel_size, activation='relu', name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
        model.add(Convolution2D(128, kernel_size, activation='relu', name='block2_conv1'))
        model.add(Convolution2D(128, kernel_size, activation='relu', name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(320, activation='relu', name='fc1'))
        model.add(Dense(192, activation='relu', name='fc2'))
        model.add(Dense(nb_classes, activation='softmax', name='predictions'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
        model.save(model_local)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    else:
        model = load_model(model_local)
        test_score = model.evaluate(x_test, y_test, verbose=0)
        train_score = model.evaluate(x_train, y_train, verbose=0)
        print('test_score:', test_score)
        print('train_score:', train_score)
        print('svhn test loaded')


if __name__ == '__main__':
    main('../original_model/svhn_original.h5')