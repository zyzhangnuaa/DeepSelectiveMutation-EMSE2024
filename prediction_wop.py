import csv
import os
import glob
import argparse
from collections import defaultdict
from utils import load_mnist, load_cifar10, load_svhn
import numpy as np
import keras.backend as K
from tensorflow.keras.models import load_model
import gc
from keras.datasets import mnist, cifar10
import pandas as pd
from redundancy_analysis import write_list_to_csv


def inif_dict():
    mutant_class_input_killing_info = defaultdict(list)  # 记录所有变异体的杀死情况
    for i in range(10):
        mutant_class_input_killing_info[i] = []
    return mutant_class_input_killing_info


# 对除去变异算子op产生的变异体进行预测
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
                        default='mutated_model_all',
                        help='mutated model saved path')
    parser.add_argument('-predictions_path',
                        type=str,
                        default='predictions_all',
                        help='predictions path of all mutant')
    parser.add_argument('-dataset',
                        type=str,
                        default='mnist',
                        help='mnist or cifar10')
    parser.add_argument('-op',
                        type=str,
                        default='GF',
                        help='do not consider the mutants of op')

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    predictions_path = args.predictions_path
    dataset = args.dataset
    op = args.op

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = y_test.flatten()
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
        y_test = y_test.flatten()

    # 原始模型上的预测结果
    original_predictions = os.path.join(predictions_path, subject_name, 'result0', 'original', 'orig_test.npy')
    ori_predict = np.load(original_predictions)

    correct_index = np.where(ori_predict == y_test)[0]

    mutants_predctions = os.path.join(predictions_path, subject_name, 'result0', 'mutant', 'test')

    all_mutant_class_input_killing_info = {}

    mutants_path = glob.glob(os.path.join(mutated_model, subject_name, '*.h5'))
    mutants_num = len(mutants_path)
    print('mutants_num:', mutants_num)
    all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    all_killed_input_list_of_mutant = {}  # 记录杀死每个变异体的测试输入
    unkilled_all_mutant_class_killing_info = {}  # 记录所有变异体的杀死情况
    killed_list = []
    unkilled_list = []
    for mutant in mutants_path:
        mutant_name = (mutant.split("\\"))[-1].replace(".h5", "")
        if mutant_name.split('_')[1] == op:
            print('不考虑' + op)
            continue
        mutant_predctions_path = os.path.join(mutants_predctions, mutant_name + "_test.npy")
        mutant_class_killing_info = set()  # 记录单个变异体的杀死情况
        mutant_class_input_dict = inif_dict()  # 记录每个变异体上杀死每个类别的输入
        killed_input_list_of_mutant = []

        result = np.load(mutant_predctions_path)

        killing_inputs = np.where(y_test[correct_index] != result[correct_index])[0]

        if len(killing_inputs) != 0:
            killed_list.append(mutant_name)
        else:
            unkilled_list.append(mutant_name)

        for index in correct_index:
            if ori_predict[index] != result[index]:
                killed_input_list_of_mutant.append(index)
                mutant_class_killing_info.add(ori_predict[index])
                mutant_class_input_dict[ori_predict[index]].append(index)
        all_mutant_class_input_killing_info[mutant_name] = mutant_class_input_dict
        unkilled_mutant_class_killing_info = set(range(10)) - mutant_class_killing_info
        all_mutant_class_killing_info[mutant_name] = mutant_class_killing_info
        unkilled_all_mutant_class_killing_info[mutant_name] = unkilled_mutant_class_killing_info
        all_killed_input_list_of_mutant[mutant_name] = killed_input_list_of_mutant

    killed_num = 0
    unkilled_num = 0
    for i in all_mutant_class_killing_info.keys():
        killed_num += len(all_mutant_class_killing_info[i])
        unkilled_num += len(unkilled_all_mutant_class_killing_info[i])

    print('总的变异体-类别对的数量为:%s, 杀死的数量为:%s' % (mutants_num * 10, killed_num))
    print('总的变异体-类别对的数量为:%s, 未杀死的数量为:%s' % (mutants_num * 10, unkilled_num))
    print('总的变异体的数量为:%s, 杀死的数量为:%s, 未杀死的数量为:%s' % (mutants_num, len(killed_list), len(unkilled_list)))

    predictions_path_wop = os.path.join('predictions_wop', 'W' + op, subject_name)
    if not os.path.exists(predictions_path_wop):
        try:
            os.makedirs(predictions_path_wop)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    killed_csv_file = os.path.join(predictions_path_wop, "killed_class.csv")
    unkilled_csv_file = os.path.join(predictions_path_wop, "unkilled_class.csv")
    mutant_class_input_info_path = os.path.join(predictions_path_wop, "class_input_info.csv")
    killed_mutant_csv_file = os.path.join(predictions_path_wop, "killed_mutant.csv")
    unkilled_mutant_csv_file = os.path.join(predictions_path_wop, "unkilled_mutant.csv")
    killed_input_list_of_mutant_csv_file = os.path.join(predictions_path_wop, "killed_input_of_mutant.csv")

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

    with open(killed_input_list_of_mutant_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for key, value in all_killed_input_list_of_mutant.items():
            writer.writerow([key, value])

    pd.DataFrame(all_mutant_class_input_killing_info).to_csv(mutant_class_input_info_path, index=False)

    write_list_to_csv(killed_mutant_csv_file, killed_list)
    write_list_to_csv(unkilled_mutant_csv_file, unkilled_list)


if __name__ == '__main__':
    prediction()




