import glob
import os
import numpy as np
import h5py
import pandas as pd

from calculate_ms import read_set_from_csv, related_class
from redundancy_analysis import reader_list_from_csv


def kill_class_num_op(class_dict, op):
    class_sum_dict = {}
    for mutant in class_dict:
        if mutant.split('_')[1] == op:
            killed_class = class_dict[mutant]
            for c in killed_class:
                try:
                    class_sum_dict[c] += 1
                except Exception as e:
                    class_sum_dict[c] = 1
    return class_sum_dict


def kill_num_of_t(index_class, class_dict, all_mutant_class_input_killing_info, op):
    all_kill_num_dict = {}
    for c in range(10):
        kill_num_dict = {}
        index = index_class[c]
        for i in index:
            kill_num_dict[i] = 0
        for mutant in class_dict.keys():
            if mutant.split('_')[1] != op:
                continue
            kill_test = set(list(eval(all_mutant_class_input_killing_info[mutant][c])))
            for i in index:
                if i in kill_test:
                    kill_num_dict[i] += 1
        all_kill_num_dict[c] = kill_num_dict
    return all_kill_num_dict


# 计算每个变异体-类别对的质量分数Qm-c
def calculate_qm(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op):
    # 计算变异算子的质量分数
    # 首先计算每个变异体在每个类上的质量分数
    # 然后计算每个变异体上的质量分数
    # 最后计算变异算子的质量分数
    index_class = {}  # 记录每个类别上的测试充分集             |T|
    hf = h5py.File(test_adequacy_set_save_path, 'r')
    for i in range(10):
        index_class[i] = np.asarray(hf.get('index_' + str(i)))
    # print('index_class:', index_class)
    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)

    # 变异算子op的变异体在每个类别上可以被杀死的类别数           |M|-|E|
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    class_sum_dict = kill_class_num_op(related_class_dict, op)
    # print('class_sum_dict:', class_sum_dict)

    # 杀死每个变异体类别对的输入
    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    # 统计类别c的测试充分集中的每个测试输入在指定类别上杀死的类别数   {0：{1987:3, 622:2}, 1:{546:2}}
    all_kill_num_dict = kill_num_of_t(index_class, related_class_dict, all_mutant_class_input_killing_info, op)
    # print('all_kill_num_dict:', all_kill_num_dict)

    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))

    # 计算每个变异体-类别对的质量分数Qm-c
    mutant_class_dict = {}  # mutant_class_dict = {m1:{0:1,1:0.5},m2:{0:0,1:0.8}}
    mutant_dict = {}  # mutant_dict = {m1:0.6, m2:0.8}
    # mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    # for mutant in mutants_path:
    #     mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
    for mutant_name in mutant_list:
        if mutant_name.split('_')[1] != op:
            continue
        mutant_dict[mutant_name] = 0
        class_dict = {}
        num_not_0 = 0
        for i in range(10):
            kill_test = set(list(eval(all_mutant_class_input_killing_info[mutant_name][i]))).intersection(set(index_class[i]))
            print('kill_test_' + str(i) + ':', kill_test)
            if len(kill_test) == 0:
                class_dict[i] = 0
            else:
                sum = 0
                num_not_0 += 1
                for t in kill_test:
                    sum += all_kill_num_dict[i][t]
                class_dict[i] = 1-(1/(class_sum_dict[str(i)] * len(index_class[i]))) * sum
                mutant_dict[mutant_name] += class_dict[i]
            mutant_class_dict[mutant_name] = class_dict

        mutant_dict[mutant_name] /= num_not_0

    # print(mutant_dict)

    # 每个变异算子的质量分数
    op_qs = {}
    op_num = {}
    for mutant in mutant_dict.keys():
        op = mutant.split('_')[1]
        try:
            op_qs[op] += mutant_dict[mutant]
            op_num[op] += 1
        except Exception as e:
            op_qs[op] = mutant_dict[mutant]
            op_num[op] = 1
    for op in op_qs:
        # op_qs[op] /= op_num[op]
        op_qs[op] = round(op_qs[op] / op_num[op], 4)
    return op_qs


def sort_dict(d, reverse):
    d_order = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    print(d_order)


if __name__ == '__main__':
    subject_name = 'lenet5'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'
    # --计算变异算子的质量分数--
    # 首先计算每个变异体在每个类上的质量分数
    # 然后计算每个变异体上的质量分数
    # 最后计算变异算子的质量分数
    op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # for lenet5 mnist svhn
    # op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs', 'LR'] # for cifar10
    op_qs_dict = {}
    for op in op_list:
        num = 0
        test_adequacy_set_save_path = os.path.join(
            os.path.join('test_suit', op + '_class', subject_name, 'index_' + str(num) + '.h5'))
        op_qs = calculate_qm(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op)
        op_qs_dict[op] = op_qs[op]
    print(op_qs_dict)
    sort_dict(op_qs_dict, True)