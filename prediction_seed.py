import csv
import importlib
import os
import glob
import argparse
import h5py
from collections import defaultdict
from utils import load_mnist, load_cifar10, load_svhn
from tensorflow.keras.models import load_model
import numpy as np
import keras.backend as K
import gc
from keras.datasets import mnist, cifar10
import pandas as pd
from redundancy_analysis import write_list_to_csv
from keras.layers import Input
from utils import load_new_data
from tensorflow.keras.models import load_model


def inif_dict():
    mutant_class_input_killing_info = defaultdict(list)  # 记录所有变异体的杀死情况
    for i in range(10):
        mutant_class_input_killing_info[i] = []
    return mutant_class_input_killing_info


def prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='cifar10',
                        help='subject name')
    # 原始模型训练参数
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_all',
                        help='mutated model saved path')
    parser.add_argument('-predictions_path',
                        type=str,
                        default='predictions_all',
                        help='predictions_path')
    parser.add_argument('-dataset',
                        type=str,
                        default='cifar10',
                        help='mnist or cifar10 or svhn')
    parser.add_argument('-test_set_kind',
                        type=str,
                        default='random',
                        help='seed or random or max')
    parser.add_argument('-random_time',
                        type=int,
                        default=4,
                        help='random time')
    # parser.add_argument('-data_path',
    #                     type=str,
    #                     default='new_inputs/lenet5/generated_inputs',
    #                     help='new input path')

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    test_set_kind = args.test_set_kind
    predictions_path = args.predictions_path
    # data_path = args.data_path
    dataset = args.dataset
    random_time = args.random_time

    if subject_name == 'lenet5':
        op_list = ['NS', 'DM', 'NAI', 'LE', 'DF', 'AFRs', 'DR']  # for lenet5
    elif subject_name == 'mnist':
        op_list = ['NAI', 'LE', 'LAs', 'DR', 'AFRs', 'DM', 'DF']  # for mnist
    elif subject_name == 'svhn':
        op_list = ['NS', 'NEB', 'GF', 'WS', 'NAI', 'LAa', 'DM']  # for svhn
    elif subject_name == 'cifar10':
        op_list = ['GF', 'NEB', 'NS', 'WS', 'NAI', 'LAa', 'LR']  # for cifar10

    if dataset == 'mnist':
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
    else:
        img_rows, img_cols = 32, 32
        input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    if test_set_kind == 'seed':
        data_path = os.path.join('seed_input', subject_name, 'seeds_100')
        x_test, y_test = load_new_data(data_path, dataset)
        # x_test = np.concatenate((x_test, x_test_mki), axis=0)
        # y_test = np.concatenate((y_test, y_test_mki), axis=0)
        print('x_test_shape', x_test.shape)
        print('y_test_shape', y_test.shape)
    elif test_set_kind == 'random':
        data_path = os.path.join('seed_input', subject_name, 'seeds_100_' + str(random_time))
        test_set_kind = test_set_kind + str(random_time)
        x_test, y_test = load_new_data(data_path, dataset)
        print('x_test_shape', x_test.shape)
        print('y_test_shape', y_test.shape)
    elif test_set_kind == 'max':
        data_path = os.path.join('seed_input', subject_name, 'seeds_max_100')
        x_test, y_test = load_new_data(data_path, dataset)
        print('x_test_shape', x_test.shape)
        print('y_test_shape', y_test.shape)

    # 在原始模型上进行预测
    original_model_path = os.path.join(original_model, subject_name + '_original.h5')
    original_predictions = os.path.join(predictions_path, subject_name, 'result0', 'original', 'orig_' + test_set_kind + '.npy')

    if not os.path.exists(os.path.join(predictions_path, subject_name, 'result0', 'original')):
        try:
            os.makedirs(os.path.join(predictions_path, subject_name, 'result0', 'original'))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    if not os.path.exists(original_predictions):
        ori_model = load_model(original_model_path)
        ori_predict = ori_model.predict(x_test).argmax(axis=-1)

        np.save(original_predictions, ori_predict)
    else:
        ori_predict = np.load(original_predictions)
        print('ori_predict:', ori_predict)
        print('y_test:', y_test)

    correct_index = np.where(ori_predict == y_test)[0]
    print('correct_index:', correct_index)
    print(len(correct_index))

    mutants_predctions = os.path.join(predictions_path, subject_name, 'result0', 'mutant', test_set_kind)
    if not os.path.exists(mutants_predctions):
        try:
            os.makedirs(mutants_predctions)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    all_mutant_class_input_killing_info = {}

    mutants_path = glob.glob(os.path.join(mutated_model, subject_name, '*.h5'))
    mutants_num = 0
    all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    unkilled_all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    killed_list = []
    unkilled_list = []
    for mutant in mutants_path:
        mutant_name = (mutant.split("\\"))[-1].replace(".h5", "")
        op = mutant_name.split('_')[1]
        if op not in op_list:
            continue
        mutants_num += 1
        mutant_predctions_path = os.path.join(mutants_predctions, mutant_name + "_" + test_set_kind + ".npy")
        mutant_class_killing_info = set()  # 记录单个变异体的杀死情况
        mutant_class_input_dict = inif_dict()  # 记录每个变异体上杀死每个类别的输入
        if not os.path.exists(mutant_predctions_path):
            model = load_model(mutant)
            result = model.predict(x_test).argmax(axis=-1)
            np.save(mutant_predctions_path, result)
            K.clear_session()
            del model
            gc.collect()
        else:
            result = np.load(mutant_predctions_path)
            if mutant_name == 'lenet5_GF_0.01_mutated_1':
                print('cb:', result)

        killing_inputs = np.where(y_test[correct_index] != result[correct_index])[0]

        if len(killing_inputs) != 0:
            killed_list.append(mutant_name)
        else:
            unkilled_list.append(mutant_name)

        for index in correct_index:
            if ori_predict[index] != result[index]:
                mutant_class_killing_info.add(ori_predict[index])
                mutant_class_input_dict[ori_predict[index]].append(index)
        all_mutant_class_input_killing_info[mutant_name] = mutant_class_input_dict
        unkilled_mutant_class_killing_info = set(range(10)) - mutant_class_killing_info
        all_mutant_class_killing_info[mutant_name] = mutant_class_killing_info
        unkilled_all_mutant_class_killing_info[mutant_name] = unkilled_mutant_class_killing_info

    killed_num = 0
    unkilled_num = 0
    for i in all_mutant_class_killing_info.keys():
        killed_num += len(all_mutant_class_killing_info[i])
        unkilled_num += len(unkilled_all_mutant_class_killing_info[i])

    print('mutants_num:', mutants_num)

    print('总的变异体-类别对的数量为:%s, 杀死的数量为:%s' % (mutants_num * 10, killed_num))
    print('总的变异体-类别对的数量为:%s, 未杀死的数量为:%s' % (mutants_num * 10, unkilled_num))
    print('总的变异体的数量为:%s, 杀死的数量为:%s, 未杀死的数量为:%s' % (mutants_num, len(killed_list), len(unkilled_list)))

    killed_csv_file = os.path.join(mutants_predctions, "killed_class.csv")
    unkilled_csv_file = os.path.join(mutants_predctions, "unkilled_class.csv")
    mutant_class_input_info_path = os.path.join(mutants_predctions, "class_input_info.csv")
    killed_mutant_csv_file = os.path.join(mutants_predctions, "killed_mutant.csv")
    unkilled_mutant_csv_file = os.path.join(mutants_predctions, "unkilled_mutant.csv")

    with open(killed_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['mutant', 'killed_class'])
        for key, value in all_mutant_class_killing_info.items():
            writer.writerow([key, value])
        writer.writerow(['all_mutant_class_pair', mutants_num * 10])
        writer.writerow(['killed_num', killed_num])

    with open(unkilled_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['mutant', 'unkilled_class'])
        for key, value in unkilled_all_mutant_class_killing_info.items():
            writer.writerow([key, value])
        writer.writerow(['all_mutant_class_pair', mutants_num * 10])
        writer.writerow(['unkilled_num', unkilled_num])

    pd.DataFrame(all_mutant_class_input_killing_info).to_csv(mutant_class_input_info_path, index=False)

    write_list_to_csv(killed_mutant_csv_file, killed_list)
    write_list_to_csv(unkilled_mutant_csv_file, unkilled_list)


if __name__ == '__main__':
    prediction()




