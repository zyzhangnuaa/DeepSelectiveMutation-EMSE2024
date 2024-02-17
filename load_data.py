import glob
import importlib
import os
import subprocess
import h5py
import numpy as np
import math
from collections import defaultdict
from keras.datasets import mnist, cifar10
from tensorflow.python.keras import Input
from keras.utils import to_categorical

# model.summary()
# acc_train = model.evaluate(x_train, y_train, verbose=0)[1]
# acc_test = model.evaluate(x_test, y_test, verbose=0)[1]
# print(acc_train)
# print(acc_test)

# size = 0
# size_list = []
# for i in range(5):
#     hf = h5py.File(os.path.join('test_suit', 'dl_class', 'lenet5', 'data_' + str(i) + '.h5'), 'r')
#     x_test = np.asarray(hf.get('x_test'))
#     y_test = np.asarray(hf.get('y_test'))
#     size += x_test.shape[0]
#     size_list.append(x_test.shape[0])
# print('size_list:', size_list)
# print('arg_size:', size/len(size_list))
# print('std:', np.std(size_list))
#
# size = 0
# size_list = []
# for i in range(20):
#     hf = h5py.File(os.path.join('test_suit', 'random_class_rs', 'lenet5', 'index_' + str(i) + '.h5'), 'r')
#     x_test = np.asarray(hf.get('x_test'))
#     y_test = np.asarray(hf.get('y_test'))
#     size += x_test.shape[0]
#     size_list.append(x_test.shape[0])
# print('size_list:', size_list)
# print('arg_size:', size/len(size_list))
# print('std:', np.std(size_list))

# d = len(np.load('predictions_dl_fc/cifar10/result0/mutant/sample_random_mutant/cifar10_GF_0.003_mutated_0_sample_random_mutant.npy'))
# print(d)
# d = len(np.load('predictions_dl_fc/cifar10/result1/mutant/sample_random_mutant/cifar10_GF_0.003_mutated_0_sample_random_mutant.npy'))
# print(d)
# hf = h5py.File(os.path.join('test_suit', 'cifar10.h5'), 'r')
# lenet5_non_num_list = list(np.asarray(hf.get('all_kill_growth_list_non_num')))
# lenet5_non_ratio_list = list(np.asarray(hf.get('all_kill_growth_list_non_ratio')))
# lenet5_random_num_list = list(np.asarray(hf.get('all_kill_growth_list_random_num')))
# lenet5_random_ratio_list = list(np.asarray(hf.get('all_kill_growth_list_random_ratio')))
#
#
# print(lenet5_non_ratio_list)
# print(lenet5_random_ratio_list)
#
# print(len(lenet5_non_ratio_list))
# print(len(lenet5_random_ratio_list))
#
# index = [i for i in range(len(lenet5_non_ratio_list)) if lenet5_non_ratio_list[i] > lenet5_random_ratio_list[i]]
# print(index)


# hf = h5py.File('test_suit/LD_class/cifar10/data_0.h5', 'r')
# x_test = np.asarray(hf.get('x_test'))
# y_test = np.asarray(hf.get('y_test'))
# print('x_test_shape:', x_test.shape)
# print('y_test_shape:', y_test.shape)
# print(y_test)

# index_class = {}
# test_adequacy_set_save_path = 'test_suit/all_class/mnist/'
# for i in range(6):
#     hf = h5py.File(test_adequacy_set_save_path + 'index_' + str(i) + '.h5', 'r')
#     index = np.asarray(hf.get('index'))
#     print('index_class:', index)

# all class lenet5:123  mnist:119  cifar10:93  svhn:95
# reduce_class lenet5:109   mnist:112  cifar10:87  svhn:93
#                   88.62%   94.12%    93.55%   97.89%

subject_name = 'lenet5'
if subject_name == 'lenet5':
    all = 123
elif subject_name == 'mnist':
    all = 119
elif subject_name == 'cifar10':
    all = 93
elif subject_name == 'svhn':
    all = 95
index_class = {}
test_adequacy_set_save_path = 'test_suit_newexp/random_seg_class_qs/lenet5/'
sum = 0
index_list = []
for i in range(120,140):
    hf = h5py.File(test_adequacy_set_save_path + 'index_' + str(i) + '.h5', 'r')
    index = np.asarray(hf.get('index'))
    index_list.append(len(index))
    sum += len(index)
    print('index_class:', len(index))
arg = round(sum/5,1)
# print('arg:', arg)
print(index_list)
test_lost_list = []
for index in index_list:
    test_lost_list.append(round((all - index) / all * 100, 2))
print(test_lost_list)
print('test lost_avg:', round((all - arg) / all * 100, 2))
# print(np.std(index_list))


