import argparse
import time

import tensorflow as tf
import numpy as np
import keras
import random, math
from tensorflow.keras.models import load_model
from progressbar import ProgressBar
import gc
import keras.backend as K
import utils_source_op
import network
from network import Lenet5, Mnist, Cifar10, Svhn
import source_level_operators
import os
from utils_model_op import color_preprocessing, model_predict
from keras.datasets import mnist, cifar10
from source_level_operators import source_level_operator_name
from utils import load_svhn

utils = utils_source_op.GeneralUtils()
source_mut_opts = source_level_operators.SourceMutationOperators()


def generator():
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='svhn',
                        help='subject name')
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_all',
                        help='mutated model saved path')
    parser.add_argument('-model_nums',
                        type=int,
                        default=20)
    parser.add_argument('-dataset',
                        type=str,
                        default="svhn",
                        help="mnist or cifar10 or svhn")
    parser.add_argument('-operator',
                        type=int,
                        default=7,
                        help="mutation operator 0-DR 1-LE 2-DM 3-DF 4-NP 5-LR 6-LAs 7 AFRs")
    parser.add_argument('-ratio',
                        type=float,
                        default=0.01,
                        help="ratio of important neurons to be mutated")
    parser.add_argument('-threshold',
                        type=float,
                        default=0.9,
                        help="ori acc * threshold must > mutants acc")

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    model_nums = args.model_nums
    dataset = args.dataset
    operator = args.operator
    mutation_ratio = args.ratio
    threshold = args.threshold

    # load data
    if dataset == 'mnist':
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train, x_test = color_preprocessing(x_train, x_test, 0, 255)
        # x_test = x_test.reshape(len(x_test), 28, 28, 1)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif dataset == 'cifar10':
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # x_train, x_test = color_preprocessing(x_train, x_test, [125.307, 122.95, 113.865], [62.9932, 62.0887, 66.7048])
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()

    mutated_model_path = os.path.join(mutated_model, subject_name, '')
    if not os.path.exists(mutated_model_path):
        try:
            os.makedirs(mutated_model_path)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    original_model_path = os.path.join(original_model, subject_name + '_original')
    ori_model_path = original_model_path + '.h5'
    ori_model = load_model(ori_model_path)
    ori_acc = model_predict(ori_model, x_test, y_test)
    threshold = ori_acc * threshold
    print('ori_acc:', ori_acc)
    print('threshold:', threshold)

    # mutants generation
    p_bar = ProgressBar().start()
    i = 1
    start_time = time.clock()
    while i <= model_nums:
        model_save_path = subject_name + '_' + source_level_operator_name(operator) + '_' + str(mutation_ratio) + '_mutated_' + str(i) + '.h5'
        if not os.path.exists(mutated_model_path + model_save_path):

            if subject_name == 'lenet5':
                network = Lenet5()
                model = network.creat_lenet5_model()
                (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
                print('train_datas shape:', train_datas.shape)
                print('train_labels shape:', train_labels.shape)
                print('test_datas shape:', test_datas.shape)
                print('test_labels shape:', test_labels.shape)
            elif subject_name == 'mnist':
                network = Mnist()
                model = network.creat_mnist_model()
                (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
                print('train_datas shape:', train_datas.shape)
                print('train_labels shape:', train_labels.shape)
                print('test_datas shape:', test_datas.shape)
                print('test_labels shape:', test_labels.shape)
            elif subject_name == 'cifar10':
                network = Cifar10()
                model = network.creat_cifar10_model()
                (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
                print('train_datas shape:', train_datas.shape)
                print('train_labels shape:', train_labels.shape)
                print('test_datas shape:', test_datas.shape)
                print('test_labels shape:', test_labels.shape)
            elif subject_name == 'svhn':
                network = Svhn()
                model = network.creat_svhn_model()
                (train_datas, train_labels), (test_datas, test_labels) = network.load_data()
                print('train_datas shape:', train_datas.shape)
                print('train_labels shape:', train_labels.shape)
                print('test_datas shape:', test_datas.shape)
                print('test_labels shape:', test_labels.shape)

            if source_level_operator_name(operator) == 'DR':  # 数据重复
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DR_mut((train_datas, train_labels), model,
                                                                                             mutation_ratio)
                utils.print_messages_SMO('DR', train_datas=train_datas, train_labels=train_labels,
                                     mutated_datas=mutated_datas, mutated_labels=mutated_labels,
                                     mutation_ratio=mutation_ratio)
            elif source_level_operator_name(operator) == 'LE':  # 标签错误
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LE_mut((train_datas, train_labels), model,
                                                                                             0, 9, mutation_ratio)
                mask_equal = mutated_labels == train_labels
                mask_equal = np.sum(mask_equal, axis=1) == 10
                count_diff = len(train_labels) - np.sum(mask_equal)
                print(len(train_labels))
                print('Mutation ratio:', mutation_ratio)
                print('Number of mislabeled labels:', count_diff)

            elif source_level_operator_name(operator) == 'DM':  # 数据缺失
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DM_mut((train_datas, train_labels), model,
                                                                                             mutation_ratio)
                utils.print_messages_SMO('DM', train_datas=train_datas, train_labels=train_labels,
                                         mutated_datas=mutated_datas, mutated_labels=mutated_labels,
                                         mutation_ratio=mutation_ratio)

            elif source_level_operator_name(operator) == 'DF':  # 数据打乱
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.DF_mut((train_datas, train_labels), model,
                                                                                             mutation_ratio)
            elif source_level_operator_name(operator) == 'NP':  # 添加噪声
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.NP_mut((train_datas, train_labels), model,
                                                                                             mutation_ratio, STD=100)
            elif source_level_operator_name(operator) == 'LR':  # 层移除
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LR_mut((train_datas, train_labels), model,
                                                                                             mutated_layer_indices=None)
            elif source_level_operator_name(operator) == 'LAs':  # 层增加
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.LAs_mut((train_datas, train_labels), model,
                                                                                              mutated_layer_indices=None)
            elif source_level_operator_name(operator) == 'AFRs':  # 激活函数移除
                (mutated_datas, mutated_labels), mutated_model = source_mut_opts.AFRs_mut((train_datas, train_labels), model,
                                                                                               mutated_layer_indices=None)

            new_model = network.train_model(mutated_model, mutated_datas, mutated_labels)
            new_model.summary()
            new_acc = model_predict(new_model, x_test, y_test)
            print('new_acc:', new_acc)

            if new_acc < threshold:
                K.clear_session()
                del new_model
                gc.collect()
                continue
            new_model.save(mutated_model_path + model_save_path)
            p_bar.update(int((i / model_nums) * 100))
            i += 1
            K.clear_session()
            del new_model
            gc.collect()

    p_bar.finish()
    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)


if __name__ == '__main__':
    generator()

