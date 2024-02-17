import glob
import os
import numpy as np
import h5py
import pandas as pd
import csv
from calculate_ms import read_set_from_csv, related_class
from redundancy_analysis import reader_list_from_csv
from calculate_ms_all_class import print_test_kill_class


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

    # 计算每个变异体-类别对的质量分数Qm-c
    mutant_class_dict = {}  # mutant_class_dict = {m1:{0:1,1:0.5},m2:{0:0,1:0.8}}
    mutant_dict = {}  # mutant_dict = {m1:0.6, m2:0.8}
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    for mutant in mutants_path:
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        if mutant_name.split('_')[1] != op:
            continue
        mutant_dict[mutant_name] = 0
        class_dict = {}
        for i in range(10):
            kill_test = set(list(eval(all_mutant_class_input_killing_info[mutant_name][i]))).intersection(set(index_class[i]))
            # print('kill_test_' + str(i) + ':', kill_test)
            if len(kill_test) == 0:
                class_dict[i] = 0
            else:
                sum = 0
                for t in kill_test:
                    sum += all_kill_num_dict[i][t]
                class_dict[i] = 1-(1/(class_sum_dict[str(i)] * len(index_class[i]))) * sum
                mutant_dict[mutant_name] += class_dict[i]
            mutant_class_dict[mutant_name] = class_dict

    for mutant in mutant_dict.keys():
        mutant_dict[mutant] /= 10
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
        op_qs[op] = round(op_qs[op] / op_num[op], 4)
    return op_qs


# 计算每个变异体-类别对的质量分数Qm-c
def calculate_qm_non(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op):
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
            # print('kill_test_' + str(i) + ':', kill_test)
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
        op_qs[op] = round(op_qs[op] / op_num[op], 4)
    return op_qs


def sort_dict(d, reverse):
    d_order = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    return d_order


def median(value):
    value.sort()
    length = len(value)
    if length % 2 == 0:
        return round((value[int(length/2) - 1] + value[int(length/2)])/2, 4)
    else:
        return value[int(length/2)]


if __name__ == '__main__':
    subject_name = 'svhn'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'
    # --计算变异算子的质量分数--
    # 首先计算每个变异体在每个类上的质量分数
    # 然后计算每个变异体上的质量分数
    # 最后计算变异算子的质量分数
    test_kill_class = print_test_kill_class(subject_name, mutated_model_path, predictions_path)
    print('op_test_kill_class_num:', test_kill_class)

    # 所有的变异算子
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    op_set = set()
    for mutant in mutants_path:
        op_name = (mutant.split("\\"))[-1].replace(".h5", "").split('_')[1]
        op_set.add(op_name)
    op_list = list(op_set)
    print('op_list:', op_list)
    print(len(op_list))

    all_op_qs_dict = {}  # 计算Qo
    all_op_qs_dict_non = {}   # 计算Qo'
    run_time = 5
    for time in range(run_time):
        op_qs_dict = {}
        op_qs_dict_non = {}
        for op in op_list:
            test_adequacy_set_save_path = os.path.join(
                os.path.join('test_suit', op + '_class', subject_name, 'index_' + str(time) + '.h5'))

            # 计算每个变异算子的质量分数Qo
            op_qs = calculate_qm(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op)
            op_qs_dict[op] = op_qs[op]

            # 计算每个变异算子的质量分数Qo
            op_qs_non = calculate_qm_non(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op)
            try:
                op_qs_dict_non[op] = op_qs_non[op]
            except Exception as e:
                print('aa')

            try:
                all_op_qs_dict[op] += op_qs_dict[op]
                try:
                    all_op_qs_dict_non[op] += op_qs_dict_non[op]
                except Exception as e:
                    print('aa')
            except Exception as e:
                all_op_qs_dict[op] = op_qs_dict[op]
                try:
                    all_op_qs_dict_non[op] = op_qs_dict_non[op]
                except Exception as e:
                    print('aa')

    arg_op_qs_dict = {k: round(v/run_time, 2) for k, v in all_op_qs_dict.items()}
    print('arg_op_qs_dict:', arg_op_qs_dict)
    arg_op_qs_dict_non = {k: round(v / run_time, 2) for k, v in all_op_qs_dict_non.items()}
    print('arg_op_qs_dict_non:', arg_op_qs_dict_non)

    e = {}
    for key in test_kill_class.keys():
        if key != 'all':
            e[key] = round((200 - test_kill_class[key]) / 200, 2)
    print('e_rate:', e)

    e_rate_sort = dict(sort_dict(e, False))
    arg_op_qs_dict_sort = dict(sort_dict(arg_op_qs_dict, True))
    print('arg_op_qs_dict_sort:', arg_op_qs_dict_sort)
    arg_op_qs_dict_sort_non = dict(sort_dict(arg_op_qs_dict_non, True))
    print('arg_op_qs_dict_sort_non:', arg_op_qs_dict_sort_non)

    # 存储质量分数
    unred_csv_file = os.path.join(predictions_path, subject_name, "qs_2.csv")

    with open(unred_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for key, value in arg_op_qs_dict_sort_non.items():
            writer.writerow([key, value])

    for key in e_rate_sort.keys():
        print(key, e_rate_sort[key], arg_op_qs_dict_sort[key], arg_op_qs_dict_sort_non[key])

    value = list(arg_op_qs_dict.values())
    mid = median(value)
    print('mid:', mid)
    value_non = list(arg_op_qs_dict_non.values())
    mid_non = median(value_non)
    print('mid_non:', mid_non)

    # 根据中位数获取要删除的变异算子  删除质量分数小于中位数的变异算子
    delete_op = sort_dict({k: v for k, v in arg_op_qs_dict.items() if v < mid}, False)
    print('delete_op:', delete_op)
    delete_op_non = sort_dict({k: v for k, v in arg_op_qs_dict_non.items() if v < mid_non}, False)
    print('delete_op_non:', delete_op_non)

    # 根据质量分数要删除的变异算子
    op1 = set(op[0] for op in delete_op)
    op2 = set(op[0] for op in delete_op_non)
    print('delete op:', op1.intersection(op2))

    # 根据中位数获取选择的变异算子  选择质量分数高于中位数的变异算子
    choice_op_non = sort_dict({k: v for k, v in arg_op_qs_dict_non.items() if v >= mid_non}, False)
    print('choice_op_non:', choice_op_non)

    # 获取等价率小于指定百分比的变异算子
    # rate = 0.2
    # op_list = []
    # for op in e.keys():
    #     if e[op] <= rate:
    #         op_list.append(op)
    # a = {}
    # for op in op_list:
    #     a[op] = arg_op_qs_dict_sort_non[op]
    # print('a:', a)
    # value = list(a.values())
    # mid = median(value)
    # print('a_mid:', mid)
    # class1 = {}
    # class2 = {}
    # for op in op_list:
    #     if a[op] >= mid:
    #         class1[op] = a[op]
    #     else:
    #         class2[op] = a[op]
    # print('class1:', class1)
    # print('class2:', class2)
    #
    # op_list = []
    # for op in e.keys():
    #     if e[op] > rate:
    #         op_list.append(op)
    # a = {}
    # for op in op_list:
    #     a[op] = arg_op_qs_dict_sort_non[op]
    # print('b:', a)
    # value = list(a.values())
    # mid = median(value)
    # print('b_mid:', mid)
    # class1 = {}
    # class2 = {}
    # for op in op_list:
    #     if a[op] >= mid:
    #         class1[op] = a[op]
    #     else:
    #         class2[op] = a[op]
    # print('class3:', class1)
    # print('class4:', class2)

    # for op in op_list:
    #     num = 0
    #     test_adequacy_set_save_path = os.path.join(
    #         os.path.join('test_suit', op + '_class', subject_name, 'index_' + str(num) + '.h5'))
    #     op_qs = calculate_qm(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path, op)
    #     op_qs_dict[op] = op_qs[op]
    # print(op_qs_dict)
    # sort_dict(op_qs_dict, True)