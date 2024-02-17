import tensorflow as tf
import numpy as np
import keras
from utils_model_op import load_original_model
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout
from keras.models import Model
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.models import load_model
import scipy.io as sio
from utils import load_svhn


class Lenet5():

    def __init__(self):
        self.number_of_train_data = 50000
        self.number_of_test_data = 10000
        self.resize_width = 28
        self.resize_height = 28

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
        # train_labels = train_labels[:self.number_of_train_data]
        # test_labels = test_labels[:self.number_of_test_data]
        # train_datas = train_datas[:self.number_of_train_data]
        # test_datas = test_datas[:self.number_of_test_data]

        train_datas = train_datas.reshape(train_datas.shape[0], self.resize_width, self.resize_height, 1)
        test_datas = test_datas.reshape(test_datas.shape[0], self.resize_width, self.resize_height, 1)

        train_datas = train_datas.astype('float32')
        test_datas = test_datas.astype('float32')
        train_datas /= 255
        test_datas /= 255

        # One-hot encoding the labels
        train_labels = keras.utils.to_categorical(train_labels)
        test_labels = keras.utils.to_categorical(test_labels)

        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return load_model(file_name)

    def creat_lenet5_model(self):
        nb_classes = 10
        kernel_size = (5, 5)
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)

        model = Sequential()
        model.add(Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1',
                                input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
        model.add(Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(120, activation='relu', name='fc1'))
        model.add(Dense(84, activation='relu', name='fc2'))
        model.add(Dense(nb_classes, activation='softmax', name='predictions'))

        return model

    def train_model(self, model, train_datas, train_labels, epochs=10, batch_size=256):

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        nb_classes = 10
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, nb_classes)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(learning_rate=1.0), metrics=['accuracy'])
        print('ccccbbbbb:', train_datas.shape, train_labels.shape)
        model.fit(train_datas, train_labels, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                  verbose=1)

        return model


class Mnist():

    def __init__(self):
        self.number_of_train_data = 50000
        self.number_of_test_data = 10000
        self.resize_width = 28
        self.resize_height = 28

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()

        train_datas = train_datas.reshape(train_datas.shape[0], self.resize_width, self.resize_height, 1)
        test_datas = test_datas.reshape(test_datas.shape[0], self.resize_width, self.resize_height, 1)

        train_datas = train_datas.astype('float32')
        test_datas = test_datas.astype('float32')
        train_datas /= 255
        test_datas /= 255

        # One-hot encoding the labels
        train_labels = keras.utils.to_categorical(train_labels)
        test_labels = keras.utils.to_categorical(test_labels)

        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return load_model(file_name)

    def creat_mnist_model(self):
        nb_classes = 10
        kernel_size = (3, 3)
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)

        model = Sequential()
        model.add(Convolution2D(32, kernel_size, activation='relu', padding='same', name='block1_conv1',
                                input_shape=input_shape))
        model.add(Convolution2D(32, kernel_size, activation='relu', padding='same', name='block1_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
        model.add(Convolution2D(64, kernel_size, activation='relu', padding='same', name='block2_conv1'))
        model.add(Convolution2D(64, kernel_size, activation='relu', padding='same', name='block2_conv2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='block2_pool1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(200, activation='relu', name='fc1'))
        model.add(Dense(nb_classes, activation='softmax', name='predictions'))

        return model

    def train_model(self, model, train_datas, train_labels, epochs=10, batch_size=256):

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        nb_classes = 10
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, nb_classes)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(learning_rate=1.0), metrics=['accuracy'])
        print('ccccbbbbb:', train_datas.shape, train_labels.shape)
        model.fit(train_datas, train_labels, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                  verbose=1)

        return model


class Cifar10():

    def __init__(self):
        self.number_of_train_data = 50000
        self.number_of_test_data = 10000
        self.resize_width = 32
        self.resize_height = 32

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.cifar10.load_data()

        train_datas = train_datas.reshape(train_datas.shape[0], self.resize_width, self.resize_height, 3)
        test_datas = test_datas.reshape(test_datas.shape[0], self.resize_width, self.resize_height, 3)

        train_datas = train_datas.astype('float32')
        test_datas = test_datas.astype('float32')
        train_datas /= 255
        test_datas /= 255

        # One-hot encoding the labels
        train_labels = keras.utils.to_categorical(train_labels)
        test_labels = keras.utils.to_categorical(test_labels)

        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return load_model(file_name)

    def creat_cifar10_model(self):
        nb_classes = 10
        kernel_size = (3, 3)
        img_rows, img_cols = 32, 32
        input_shape = (img_rows, img_cols, 3)

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

        return model

    def train_model(self, model, train_datas, train_labels, epochs=50, batch_size=128):

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        nb_classes = 10
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = to_categorical(y_test, nb_classes)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(learning_rate=1.0), metrics=['accuracy'])
        print('ccccbbbbb:', train_datas.shape, train_labels.shape)
        model.fit(train_datas, train_labels, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                  verbose=1)

        return model


class Svhn():

    def __init__(self):
        # self.number_of_train_data = 50000
        # self.number_of_test_data = 10000
        self.resize_width = 32
        self.resize_height = 32

    def load_data(self):

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

        train_label = (np.arange(num_labels) == train_label[:, ]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:, ]).astype(np.float32)

        return (train_data, train_label), (test_data, test_label)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return load_model(file_name)

    def creat_svhn_model(self):
        nb_classes = 10
        kernel_size = (3, 3)
        img_rows, img_cols = 32, 32
        input_shape = (img_rows, img_cols, 3)

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

        return model

    def train_model(self, model, train_datas, train_labels, epochs=12, batch_size=128):

        # the data, shuffled and split between train and test sets
        x_train, y_train, x_test, y_test = load_svhn(one_hot=True)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        print('ccccbbbbb:', train_datas.shape, train_labels.shape)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

        return model


class FCNetwork():

    def __init__(self):
        self.number_of_train_data = 50000
        self.number_of_test_data = 10000
        self.resize_width = 28
        self.resize_height = 28

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data].reshape(-1,
                                                                      self.resize_width * self.resize_height) / 255.0
        test_datas = test_datas[:self.number_of_test_data].reshape(-1, self.resize_width * self.resize_height) / 255.0

        # Standardize training samples
        mean_px = train_datas.mean().astype(np.float32)
        std_px = train_datas.std().astype(np.float32)
        train_datas = (train_datas - mean_px) / std_px

        # Standardize test samples
        mean_px = test_datas.mean().astype(np.float32)
        std_px = test_datas.std().astype(np.float32)
        test_datas = (test_datas - mean_px) / std_px

        # One-hot encoding the labels
        train_labels = keras.utils.np_utils.to_categorical(train_labels)
        test_labels = keras.utils.np_utils.to_categorical(test_labels)

        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    def create_simple_FC_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu',
                               input_shape=(self.resize_width * self.resize_height,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])

        return model

    def create_normal_FC_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu',
                               input_shape=(self.resize_width * self.resize_height,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, name_of_file=None, epochs=20, batch_size=None,
                    with_checkpoint=False):
        if with_checkpoint:
            prefix = ''
            filepath = prefix + name_of_file + '-{epoch:02d}-{loss:.4f}.h5'
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=5, save_best_only=True,
                                                         mode='min')
            callbacks_list = [checkpoint]
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
                      verbose=False)
        else:
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=None, verbose=False)
        return model

    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels)
        if mode == 'normal':
            print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100 * acc))
            print('')

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else:
            print('Mutated model by ' + mode + ' is successfully saved at', file_name)
        print('')

    def train_and_save_simply_FC_model(self, name_of_file=None, verbose=False, with_checkpoint=False):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        model = self.create_simple_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'simple_FC_model')

    def train_and_save_normal_FC_model(self, name_of_file=None, verbose=False, with_checkpoint=False):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        model = self.create_normal_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'normal_FC_model')


class CNNNetwork():

    def __init__(self):
        self.number_of_train_data = 5000
        self.number_of_test_data = 1000
        self.width_without_padding = 28
        self.height_without_pading = 28
        self.width_with_padding = 32
        self.height_with_padding = 32
        self.num_of_channels = 1

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
        # Truncate 5000 training samples and 1000 test samples apart from original dataset
        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data]
        test_datas = test_datas[:self.number_of_test_data]

        # Reshape the training and test dataset
        train_datas = train_datas.reshape(train_datas.shape[0], self.width_without_padding, self.height_without_pading,
                                          1)
        test_datas = test_datas.reshape(test_datas.shape[0], self.width_without_padding, self.height_without_pading, 1)

        # Pad on the original images, from 28 * 28 to 32 * 32
        train_datas = np.pad(train_datas, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        test_datas = np.pad(test_datas, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

        # Standardize training samples
        mean_px = train_datas.mean().astype(np.float32)
        std_px = train_datas.std().astype(np.float32)
        train_datas = (train_datas - mean_px) / (std_px)

        # Standardize test samples
        mean_px = train_datas.mean().astype(np.float32)
        std_px = train_datas.std().astype(np.float32)
        test_datas = (test_datas - mean_px) / (std_px)

        # One-hot encoding the labels
        train_labels = keras.utils.np_utils.to_categorical(train_labels)
        test_labels = keras.utils.np_utils.to_categorical(test_labels)
        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    def create_CNN_model_1(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=6,
                                kernel_size=5,
                                strides=1,
                                activation='relu',
                                input_shape=(self.width_with_padding, self.height_with_padding, self.num_of_channels)),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(filters=16,
                                kernel_size=5,
                                strides=1,
                                activation='relu',
                                input_shape=(14, 14, 6)),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=120, activation='relu'),
            keras.layers.Dense(units=84, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        return model

    def create_CNN_model_2(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(self.width_with_padding, self.height_with_padding, self.num_of_channels)),
            keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(30, 30, 32)),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(14, 14, 32)),
            keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                strides=1,
                                activation='relu',
                                input_shape=(12, 12, 32)),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=200, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax')
        ])
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, name_of_file=None, epochs=20, batch_size=None,
                    verbose=False, with_checkpoint=False):
        if with_checkpoint:
            prefix = ''
            filepath = prefix + name_of_file + '-{epoch:02d}-{loss:.4f}.h5'
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=5, save_best_only=True,
                                                         mode='min')
            callbacks_list = [checkpoint]
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
                      verbose=verbose)
        else:
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=None, verbose=verbose)
        return model

    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels)
        if mode == 'normal':
            print('Normal model accurancy: {:5.2f}%'.format(100 * acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100 * acc))
            print('')

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else:
            print('Mutated model by ' + mode + ' is successfully saved at', file_name)
        print('')

    def train_and_save_simply_CNN_model(self, name_of_file=None, verbose=False, with_checkpoint=False, model_index=1):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        if model_index == 1:
            model = self.create_CNN_model_1()
        else:
            model = self.create_CNN_model_2()

        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels, verbose=verbose, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'CNN_model' + str(model_index))