import csv
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


def inif_dict():
    mutant_class_input_killing_info = defaultdict(list)  # 记录所有变异体的杀死情况
    for i in range(10):
        mutant_class_input_killing_info[i] = []
    return mutant_class_input_killing_info


def prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='lenet5',
                        help='subject name')
    # 原始模型训练参数
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_reduced',
                        help='mutated model saved path')
    parser.add_argument('-predictions_path',
                        type=str,
                        default='predictions_reduced',
                        help='predictions_path')
    parser.add_argument('-dataset',
                        type=str,
                        default='mnist',
                        help='mnist or cifar10')
    parser.add_argument('-test_set_kind',
                        type=str,
                        default='weak',
                        help='test or test50 or test10 or test5 or weak or diff or test10non or test5non')
    parser.add_argument('-time',
                        type=int,
                        default=0,
                        help='repeat experiment time-th')
    parser.add_argument('-data_path',
                        type=str,
                        default='new_inputs/lenet5/generated_inputs_dlmutation',
                        help='new input path')

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    test_set_kind = args.test_set_kind
    predictions_path = args.predictions_path
    dataset = args.dataset
    time = args.time

    if test_set_kind == 'sample_reduced_class':
        hf = h5py.File(os.path.join('test_suit', 'reduced_class', subject_name, 'data_' + str(time) + '.h5'), 'r')
    elif test_set_kind == 'sample_random_class':
        hf = h5py.File(os.path.join('test_suit', 'random_class', subject_name, 'data_' + str(time) + '.h5'), 'r')

    x_test = np.asarray(hf.get('x_test'))
    y_test = np.asarray(hf.get('y_test'))
    if dataset == 'cifar10':
        y_test = y_test.flatten()
    print('x_test_shape:', x_test.shape)
    print('y_test_shape:', y_test.shape)

    # 在原始模型上进行预测
    original_model_path = os.path.join(original_model, subject_name + '_original.h5')
    original_predictions = os.path.join(predictions_path, subject_name, 'result' + str(time), 'original', 'orig_' + test_set_kind + '.npy')

    if not os.path.exists(os.path.join(predictions_path, subject_name, 'result' + str(time), 'original')):
        try:
            os.makedirs(os.path.join(predictions_path, subject_name, 'result' + str(time), 'original'))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    if not os.path.exists(original_predictions):
        ori_model = load_model(original_model_path)
        ori_predict = ori_model.predict(x_test).argmax(axis=-1)

        np.save(original_predictions, ori_predict)
    else:
        ori_predict = np.load(original_predictions)

    correct_index = np.where(ori_predict == y_test)[0]

    mutants_predctions = os.path.join(predictions_path, subject_name, 'result' + str(time), 'mutant', test_set_kind)
    if not os.path.exists(mutants_predctions):
        try:
            os.makedirs(mutants_predctions)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    all_mutant_class_input_killing_info = {}

    mutants_path = glob.glob(os.path.join(mutated_model, subject_name, '*.h5'))
    mutants_num = len(mutants_path)
    print('mutants_num:', mutants_num)
    all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    unkilled_all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    killed_list = []
    unkilled_list = []
    for mutant in mutants_path:
        mutant_name = (mutant.split("\\"))[-1].replace(".h5", "")
        mutant_predctions_path = os.path.join(mutants_predctions, mutant_name + "_" + test_set_kind + ".npy")
        mutant_class_killing_info = set()  # 记录单个变异体的杀死情况
        mutant_class_input_dict = inif_dict()  # 记录每个变异体上杀死每个类别的输入
        if not os.path.exists(mutant_predctions_path):
            print(mutant)
            model = load_model(mutant)
            result = model.predict(x_test).argmax(axis=-1)
            np.save(mutant_predctions_path, result)
            K.clear_session()
            del model
            gc.collect()
        else:
            result = np.load(mutant_predctions_path)

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




