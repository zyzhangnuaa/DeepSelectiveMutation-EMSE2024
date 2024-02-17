from __future__ import print_function

import os
from keras.models import Sequential
import keras
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.keras.models import load_model


def main(model_local):
    print('aaaaaaaaaaaaa', model_local)
    nb_classes = 10
    # convolution kernel size
    kernel_size = (3, 3)

    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    if not os.path.exists(model_local):
        batch_size = 128
        nb_epoch = 50
        model = Sequential()
        model.add(Convolution2D(64, kernel_size, activation='relu', name='block1_conv1', input_shape=input_shape))
        model.add(Convolution2D(64, kernel_size, activation='relu', name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
        model.add(Convolution2D(128, kernel_size, activation='relu', name='block2_conv1'))
        model.add(Convolution2D(128, kernel_size, activation='relu', name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(256, activation='relu', name='fc1'))
        model.add(Dropout(0.5, name='dropout_1'))
        model.add(Dense(256, activation='relu', name='fc2'))
        model.add(Dense(nb_classes, activation='softmax', name='predictions'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(learning_rate=1.0), metrics=['accuracy'])
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
        print('cifar10 test loaded')


if __name__ == '__main__':
    main('../original_model/cifar10_original.h5')
