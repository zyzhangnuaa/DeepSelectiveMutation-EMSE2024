
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


# 计算约减后的变异算子上的变异分数
# 获取约减后的最小测试充分集的index
# 根据变异体上的杀死输入信息class_input_info.csv获取杀死的非等价类别
# 获取约减后的变异体上杀死的非等价类别
def calculate_ms_reduced(subject_name, mutated_model_path, predictions_path, test_adequacy_set_save_path):

    hf = h5py.File(test_adequacy_set_save_path, 'r')
    index = np.asarray(hf.get('index'))
    # print('index:', index)

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    # mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'non_redundant_mutant.csv'))
    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))

    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
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
            # op_relate_killed_class_num[op] += len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_killed_class_num[op] += len(t_killed_class_dict[mutant_name])
        except Exception as e:
            # op_relate_killed_class_num[op] = len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_killed_class_num[op] = len(t_killed_class_dict[mutant_name])
        # sum_killed_num += len(t_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
        sum_killed_num += len(t_killed_class_dict[mutant_name])

    op_relate_class_num['all'] = sum_class_num
    op_relate_killed_class_num['all'] = sum_killed_num
    print('op_relate_class_num:', op_relate_class_num)
    print('op_relate_killed_class_num:', op_relate_killed_class_num)

    ms_dic = {}
    sum = 0
    count = 0
    for op in op_relate_class_num.keys():
        if op != 'all':
            try:
                ms = round(op_relate_killed_class_num[op] / op_relate_class_num[op], 4)
            except Exception as e:
                ms = 0
            ms_dic[op] = ms
            sum += ms
            count += 1
    ms_dic['arg'] = round(sum / count, 4)
    ms_dic['all'] = round(op_relate_killed_class_num['all'] / op_relate_class_num['all'], 4)
    # print('ms_dic:', ms_dic)

    return op_relate_class_num, op_relate_killed_class_num


def sort_dict(d, reverse):
    d_order = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    print(d_order)


#对随机 计算每次的变异分数
def mutation_score_random(subject_name,mutated_model_path,predictions_path, count):
    all_mutation_score = []
    all_arg = 0
    for time in range(count*20, count*20+20):
        test_adequacy_set_save_path = 'test_suit/' + 'random_class_rs/' + subject_name + '/index_' + str(time) + '.h5'
        op_relate_class_num, op_relate_killed_class_num = calculate_ms_reduced(subject_name, mutated_model_path,
                                                                               predictions_path,
                                                                               test_adequacy_set_save_path)
        ms = round(op_relate_killed_class_num['all']/op_relate_class_num['all'], 4)
        all_mutation_score.append(ms)
        all_arg += ms
    arg = round(all_arg/20, 4)
    return all_mutation_score, arg


if __name__ == '__main__':

    # 计算约减后的变异分数  不需要执行变异体
    # 在约减后的变异体上构建测试充分集后，便可以直接计算测试集在所有变异体上的变异分数
    subject_name = 'mnist'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    test_kill_class = print_test_kill_class(mutated_model_path, predictions_path, subject_name)
    print('op_test_kill_class_num:', test_kill_class)

    unreduntant_class = print_unreduntant_class(mutated_model_path, predictions_path, subject_name)
    print('op_unreduntant_class_num:', unreduntant_class)

    # count = 11
    # op_ms_by_rs = {}
    # for i in range(count):
    #     all_op_relate_class_num = {}
    #     all_op_relate_killed_class_num = {}
    #     for time in range(i*5, i*5+5):
    #         # test_adequacy_set_save_path = 'test_suit/' + 'reduced_class/' + subject_name + '/index_' + str(time) + '.h5'
    #         test_adequacy_set_save_path = 'test_suit/' + 'single_class/' + subject_name + '/index_' + str(time) + '.h5'
    #         op_relate_class_num, op_relate_killed_class_num = calculate_ms_reduced(subject_name, mutated_model_path,
    #                                                                                predictions_path,
    #                                                                                test_adequacy_set_save_path)
    #         for op in op_relate_class_num.keys():
    #             try:
    #                 all_op_relate_class_num[op] += op_relate_class_num[op]
    #                 all_op_relate_killed_class_num[op] += op_relate_killed_class_num[op]
    #             except Exception as e:
    #                 all_op_relate_class_num[op] = op_relate_class_num[op]
    #                 all_op_relate_killed_class_num[op] = op_relate_killed_class_num[op]
    #
    #         try:
    #             all_op_relate_class_num['all'] += op_relate_class_num['all']
    #             all_op_relate_killed_class_num['all'] += op_relate_killed_class_num['all']
    #         except Exception as e:
    #             all_op_relate_class_num['all'] = op_relate_class_num['all']
    #             all_op_relate_killed_class_num['all'] = op_relate_killed_class_num['all']
    #
    #     arg_ms_dic = {}
    #     sum = 0
    #     count = 0
    #     for op in all_op_relate_class_num.keys():
    #         if op != 'all':
    #             try:
    #                 ms = round(all_op_relate_killed_class_num[op] / all_op_relate_class_num[op], 4)
    #             except Exception as e:
    #                 ms = 0
    #             arg_ms_dic[op] = ms
    #             sum += ms
    #             count += 1
    #     # arg_ms_dic['arg'] = round(sum / count, 2)
    #     arg_ms_dic['arg'] = round(sum / count, 4)
    #     arg_ms_dic['all'] = round(all_op_relate_killed_class_num['all'] / all_op_relate_class_num['all'], 4)
    #     print('###############################  ' + 'reduced' + '  ###############################################')
    #     print('arg_ms_dic:', arg_ms_dic)
    #     print('##########################################################################################')
    #     op_ms_by_rs[i] = arg_ms_dic['all']
    # print(op_ms_by_rs)

    # 暂时不用
    all_op_relate_class_num = {}
    all_op_relate_killed_class_num = {}
    for time in range(0,1):
        # test_adequacy_set_save_path = 'test_suit/' + 'reduced_class/' + subject_name + '/index_' + str(time) + '.h5'
        # test_adequacy_set_save_path = 'test_suit/' + 'single_class/' + subject_name + '/index_' + str(time) + '.h5'
        # test_adequacy_set_save_path = 'test_suit/' + 'random_class_rs/' + subject_name + '/index_' + str(time) + '.h5'
        test_adequacy_set_save_path = 'test_suit_newexp/' + 'seg_class_rs/' + subject_name + '/index_' + str(time) + '.h5'
        op_relate_class_num, op_relate_killed_class_num = calculate_ms_reduced(subject_name, mutated_model_path, predictions_path,
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
            try:
                ms = round(all_op_relate_killed_class_num[op] / all_op_relate_class_num[op], 4)
            except Exception as e:
                ms = 0
            arg_ms_dic[op] = ms
            sum += ms
            count += 1
    # arg_ms_dic['arg'] = round(sum / count, 2)
    arg_ms_dic['arg'] = round(sum / count, 4)
    arg_ms_dic['all'] = round(all_op_relate_killed_class_num['all'] / all_op_relate_class_num['all'], 4)
    print('###############################  ' + 'reduced' + '  ###############################################')
    print('arg_ms_dic:', arg_ms_dic)
    print('##########################################################################################')

    # count = 0
    # all_mutation_score, arg = mutation_score_random(subject_name, mutated_model_path, predictions_path, count)
    # print(all_mutation_score)
    # print(arg)



