
import glob
import os
import csv
from redundancy_analysis import reader_list_from_csv
import h5py
import numpy as np
from calculate_ms import related_class, read_set_from_csv, print_test_kill_class, print_unreduntant_class
import pandas as pd


def killed_class(mutated_model_path, subject_name, index, all_mutant_class_input_killing_info):
    test_killed_class_dict = {}
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    for mutant in mutants_path:
        killed_class = set()
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        for i in range(10):
            kill_test = set(list(eval(all_mutant_class_input_killing_info[mutant_name][i]))).intersection(set(index))
            # print('aa:', kill_test)
            if len(kill_test) != 0:
                killed_class.add(str(i))
        test_killed_class_dict[mutant_name] = killed_class
    return test_killed_class_dict


# 计算变异算子op的冗余分数
# 获取除去op构建的最小测试充分集的index
# 根据所有变异体上的杀死输入信息class_input_info.csv获取杀死的非冗余类别的
def calculate_rs(mutated_model_path, predictions_path, test_adequacy_set_save_path):

    hf = h5py.File(test_adequacy_set_save_path, 'r')
    index = np.asarray(hf.get('index'))
    print('index_num:', len(index))

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'non_redundant_mutant.csv'))

    test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)

    # 获取测试充分集杀死的类别信息
    t_killed_class_dict = killed_class(mutated_model_path, subject_name, index, all_mutant_class_input_killing_info)

    reduntant_class_csv = os.path.join(predictions_path, subject_name, "reduntant_class.csv")
    reduntant_class = read_set_from_csv(mutated_model_path, subject_name, reduntant_class_csv)

    op_relate_class_num = {}
    op_relate_killed_class_num = {}
    sum_class_num = 0
    sum_killed_num = 0

    for mutant_name in mutant_list:
        op = mutant_name.split('_')[1]
        try:
            op_relate_class_num[op] += len(test_killed_class_dict[mutant_name])
        except Exception as e:
            op_relate_class_num[op] = len(test_killed_class_dict[mutant_name])
        sum_class_num += len(test_killed_class_dict[mutant_name])

        try:
            op_relate_killed_class_num[op] += len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            # op_relate_killed_class_num[op] += len(t_killed_class_dict[mutant_name])
        except Exception as e:
            op_relate_killed_class_num[op] = len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            # op_relate_killed_class_num[op] = len(t_killed_class_dict[mutant_name])
        sum_killed_num += len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
        # sum_killed_num += len(t_killed_class_dict[mutant_name])

    op_relate_class_num['all'] = sum_class_num
    op_relate_killed_class_num['all'] = sum_killed_num
    print('op_relate_class_num:', op_relate_class_num)
    print('op_relate_killed_class_num:', op_relate_killed_class_num)

    ms_dic = {}
    sum = 0
    count = 0
    for op in op_relate_class_num.keys():
        if op != 'all':
            ms = round(op_relate_killed_class_num[op] / op_relate_class_num[op], 4)
            ms_dic[op] = ms
            sum += ms
            count += 1
    ms_dic['arg'] = round(sum / count, 4)
    ms_dic['all'] = round(op_relate_killed_class_num['all'] / op_relate_class_num['all'], 4)
    print('ms_dic:', ms_dic)

    return op_relate_class_num, op_relate_killed_class_num


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


# 打印冗余分数
def print_rs(subject_name, mutated_model_path, predictions_path):
    test_kill_class = print_test_kill_class(mutated_model_path, predictions_path, subject_name)
    print('op_test_kill_class_num:', test_kill_class)

    unreduntant_class = print_unreduntant_class(mutated_model_path, predictions_path, subject_name)
    print('op_unreduntant_class_num:', unreduntant_class)

    op_ms = {}  # 记录想要的冗余分数
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    op_set = set()
    for mutant in mutants_path:
        op_name = (mutant.split("\\"))[-1].replace(".h5", "").split('_')[1]
        op_set.add('W' + op_name)
    op_list = list(op_set)
    print('op_list:', op_list)
    print(len(op_list))

    for aa in op_list:  # aa = 'WGF'
        all_op_relate_class_num = {}
        all_op_relate_killed_class_num = {}
        for time in range(5):
            #  'test_suit/WGF_class/mnist/index_0.h5'
            test_adequacy_set_save_path = 'test_suit/' + aa + '_class/' + subject_name + '/index_' + str(time) + '.h5'
            op_relate_class_num, op_relate_killed_class_num = calculate_rs(mutated_model_path, predictions_path,
                                                                           test_adequacy_set_save_path)
            for op in op_relate_class_num.keys():
                try:
                    all_op_relate_class_num[op] += op_relate_class_num[op]
                    all_op_relate_killed_class_num[op] += op_relate_killed_class_num[op]
                except Exception as e:
                    all_op_relate_class_num[op] = op_relate_class_num[op]
                    all_op_relate_killed_class_num[op] = op_relate_killed_class_num[op]

            try:
                all_op_relate_class_num['all'] += op_relate_class_num['all']
                all_op_relate_killed_class_num['all'] += op_relate_killed_class_num['all']
            except Exception as e:
                all_op_relate_class_num['all'] = op_relate_class_num['all']
                all_op_relate_killed_class_num['all'] = op_relate_killed_class_num['all']

        arg_ms_dic = {}
        sum = 0
        count = 0
        for op in all_op_relate_class_num.keys():
            if op != 'all':
                ms = round(all_op_relate_killed_class_num[op] / all_op_relate_class_num[op], 4)
                arg_ms_dic[op] = ms
                sum += ms
                count += 1
        # arg_ms_dic['arg'] = round(sum / count, 2)
        arg_ms_dic['arg'] = round(sum / count, 4)
        arg_ms_dic['all'] = round(all_op_relate_killed_class_num['all'] / all_op_relate_class_num['all'], 4)
        print('###############################  ' + aa + '  ###############################################')
        print('arg_ms_dic:', arg_ms_dic)
        print('##########################################################################################')

        op_ms[aa] = arg_ms_dic['all']

    return op_ms


if __name__ == '__main__':
    # 计算每个变异算子的冗余分数 不需要执行变异体
    subject_name = 'mnist'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    # 记录想要的冗余分数
    op_rs = print_rs(subject_name, mutated_model_path, predictions_path)
    print('op_rs', op_rs)

    op_rs_sort = dict(sort_dict(op_rs, False))
    print('op_ms_sort:', op_rs_sort)

    unred_csv_file = os.path.join(predictions_path, subject_name, "rs.csv")

    with open(unred_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for key, value in op_rs_sort.items():
            writer.writerow([key, value])

    value = list(op_rs.values())
    mid = median(value)
    print('mid:', mid)

    # 根据中位数获取要删除的变异算子  删除冗余分数大于中位数的变异算子
    delete_op = sort_dict({k: v for k, v in op_rs.items() if v > mid}, False)
    print('delete_op:', delete_op)
    # 根据中位数选择变异算子  选择冗余分数小于中位数的变异算子
    choice_op = sort_dict({k: v for k, v in op_rs.items() if v <= mid}, False)
    print('choice_op:', choice_op)












