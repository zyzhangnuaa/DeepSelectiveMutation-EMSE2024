import csv
import os
import glob

import h5py

from minimal_test_suit import minimal_test_suit_index_reduce_qs_seg
from calculate_ms import print_test_kill_class, print_unreduntant_class
from calculate_ms_reduced import calculate_ms_reduced
import numpy as np
import matplotlib.pyplot as plt


# 根据冗余分数构建最小测试充分集
def read_set_from_csv(rs_csv):
    op_rs_dict = {}
    with open(rs_csv) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            op_rs_dict[row[0]] = float(row[1])
    return op_rs_dict


# 使用划分好的变异算子构建的测试充分集计算不同质量分数下变异算子的测试损失
def op_Test_Lost_By_Qs(mutated_model_path, predictions_path, subject_name, count):  # count表示根据冗余分数划分的段数
    # test_kill_class = print_test_kill_class(mutated_model_path, predictions_path, subject_name)
    # print('op_test_kill_class_num:', test_kill_class)
    # unreduntant_class = print_unreduntant_class(mutated_model_path, predictions_path, subject_name)
    # print('op_unreduntant_class_num:', unreduntant_class)

    op_test_lost_by_qs = {}
    for i in range(count):
        all_test_lost = 0
        for time in range(i * 5, i * 5 + 5):
            test_adequacy_set_save_path = 'test_suit/' + 'seg_class_qs/' + subject_name + '/index_' + str(time) + '.h5'
            hf = h5py.File(test_adequacy_set_save_path, 'r')
            index = np.asarray(hf.get('index'))
            index_len = len(index)
            print('最小测试充分集的大小为:', index_len)
            if subject_name == 'lenet5':
                all = 123
            elif subject_name == 'mnist':
                all = 119
            elif subject_name == 'cifar10':
                all = 93
            elif subject_name == 'svhn':
                all = 95
            test_lost = round((all - index_len) / all * 100, 2)
            print('test lost:', test_lost)

            all_test_lost += test_lost

        arg_test_lost = round(all_test_lost/5, 2)

        print('###############################  ' + 'reduced' + '  ###############################################')
        print('arg_test_lost:', arg_test_lost)
        print('##########################################################################################')
        op_test_lost_by_qs[i] = arg_test_lost

    return op_test_lost_by_qs


if __name__ == '__main__':
    subject_name = 'lenet5'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    # 根据质量分数划分变异算子，并构建最小的测试用例集
    # mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    # op_set = set()
    # for mutant in mutants_path:
    #     op_name = (mutant.split("\\"))[-1].replace(".h5", "").split('_')[1]
    #     op_set.add(op_name)
    # if subject_name == 'cifar10':
    #     op_set.remove('LAm')
    #     op_set.remove('LD')
    # op_list = list(op_set)
    # print(op_list)
    # print(len(op_list))
    #
    # op_qs_path = os.path.join(predictions_path, subject_name, "qs.csv")
    # op_qs =read_set_from_csv(op_qs_path)
    # print(op_qs)
    #
    # op_list_choiced = []
    # qs = 0
    # for item in op_qs.items():
    #     qs = item[1]
    #     print('minimal_qs:', qs)
    #     break
    # num = 0
    # count = 1
    # all_qs = {}  # 记录所有不同的冗余分数
    # all_qs[0] = qs
    # while set(op_list_choiced) != set(op_list):
    #     op_list_choiced = []
    #     for item in op_qs.items():
    #         if item[1] >= qs:
    #             op_list_choiced.append(item[0])
    #         else:
    #             qs = item[1]
    #             all_qs[count] = qs
    #             count += 1
    #             break
    #     print('aa:', op_list_choiced)
    #     reduce_op = list(set(op_list) - set(op_list_choiced))
    #
    #     print('lllllllllllllll:', reduce_op)
    #
    #     # 使用选择的变异算子构建最小测试用例集
    #     # for n in range(num, num+5):
    #     #     kind = 'seg_class_qs'
    #     #     minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, n, reduce_op)
    #     # num += 5
    #
    # print('count:', count)  # 不同的冗余分数的数量 也就是划分的段数
    # print('all_qs:', all_qs)

    # 下面的代码用于根据冗余分数对变异算子进行划分，计算测试损失，并绘制折线图
    # count: lenet5-13 mnist-13 cifar10-9 svhn-8
    # all_qs
    # 根据冗余分数对变异算子进行划分 计算得到的测试损失
    count = {'lenet5': 13, 'mnist': 13, 'cifar10': 14, 'svhn': 13}
    all_qs = {'lenet5': {0: 0.7974, 1: 0.7628, 2: 0.7482, 3: 0.7208, 4: 0.7172, 5: 0.706, 6: 0.703,
                         7: 0.7029, 8: 0.6368, 9: 0.6278, 10: 0.6096, 11: 0.4394, 12: 0.4382},
              'mnist': {0: 0.8054, 1: 0.8014, 2: 0.7813, 3: 0.7746, 4: 0.7708, 5: 0.762, 6: 0.7376,
                        7: 0.7236, 8: 0.715, 9: 0.6886, 10: 0.6426, 11: 0.5106, 12: 0.4561},
              'cifar10':  {0: 0.8234, 1: 0.8095, 2: 0.8032, 3: 0.7811, 4: 0.7096, 5: 0.4401, 6: 0.3911, 7: 0.3395,
                           8: 0.3162, 9: 0.296, 10: 0.2897, 11: 0.2421, 12: 0.2165, 13: 0.1801},
              'svhn': {0: 0.8586, 1: 0.8545, 2: 0.8311, 3: 0.8303, 4: 0.6483, 5: 0.358, 6: 0.2659,
                       7: 0.252, 8: 0.2111, 9: 0.2024, 10: 0.1846, 11: 0.1516, 12: 0.1386}}
    # 可以直接使用之前计算的
    # all_op_test_lost_by_qs = {}
    # for key in count.keys():
    #     op_test_lost_by_qs = op_Test_Lost_By_Qs(mutated_model_path, predictions_path, key, count[key])
    #     print('subject_name:', key)
    #     print('count:', count[key])  # 根据质量分数划分的段数
    #     print('all_qs:', all_qs[key])  # 所有不同的质量分数
    #     print('op_test_lost_by_qs:', op_test_lost_by_qs)  # 根据质量分数划分变异算子计算得到的测试损失
    #     all_op_test_lost_by_qs[key] = op_test_lost_by_qs

    # all_op_test_lost_by_qs = {'lenet5': {0: 67.48, 1: 47.97, 2: 41.46, 3: 31.71, 4: 21.95, 5: 17.89,
    #                                      6: 13.01, 7: 8.94, 8: 6.5, 9: 3.25, 10: 0.0, 11: 0.0, 12: 0.0},
    #                    'mnist':  {0: 68.91, 1: 42.02, 2: 29.41, 3: 24.37, 4: 18.49, 5: 15.13, 6: 10.08,
    #                               7: 6.72, 8: 2.52, 9: 2.52, 10: 0.0, 11: 0.0, 12: 0.0},
    #                    'cifar10': {0: 51.61, 1: 32.26, 2: 20.43, 3: 9.68, 4: 7.53, 5: 6.45, 6: 4.3,
    #                                7: 1.08, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0},
    #                    'svhn': {0: 44.21, 1: 27.37, 2: 14.74, 3: 4.21, 4: 4.21, 5: 4.21,
    #                             6: 2.11, 7: 1.05, 8: 1.05, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0}}
    # print('all_qs:', all_qs)
    # print('all_op_test_lost_by_qs:', all_op_test_lost_by_qs)

    # 根据冗余分数对应的变异分数，绘制折线图
    # mu, sigma = 100, 15
    # x = mu + sigma * np.random.randn(10000)
    #
    # plt.figure(figsize=(6, 10))
    # index = all_qs['lenet5'].values()
    # plt.subplot(411)
    # # plt.plot(index, lenet5_random_ratio_list, label='random')
    # plt.plot(index, all_op_test_lost_by_qs['lenet5'].values(), label='qs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('test lost')
    # plt.title('lenet5')
    # plt.grid(True)
    # plt.legend()
    #
    # index = all_qs['mnist'].values()
    # plt.subplot(412)
    # # plt.plot(index, lenet5_random_num_list, label='random')
    # plt.plot(index, all_op_test_lost_by_qs['mnist'].values(), label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('test lost')
    # plt.title('mnist')
    # plt.legend()
    # plt.grid(True)
    #
    # index = all_qs['cifar10'].values()
    # plt.subplot(413)
    # # plt.plot(index, cifar10_random_ratio_list, label='random')
    # plt.plot(index, all_op_test_lost_by_qs['cifar10'].values(), label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('test lost')
    # plt.title('cifar10')
    # plt.legend()
    # plt.grid(True)
    #
    # index = all_qs['svhn'].values()
    # plt.subplot(414)
    # # plt.plot(index, lenet5_random_num_list, label='random')
    # plt.plot(index, all_op_test_lost_by_qs['svhn'].values(), label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('test lost')
    # plt.title('svhn')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # lenet5
    # plt.figure(figsize=(6, 10))
    # x0 = all_qs['lenet5'].values()
    # y0 = all_op_test_lost_by_qs['lenet5'].values()
    # plt.subplot(411)
    # # plt.plot(index, lenet5_random_ratio_list, label='random')
    # plt.plot(x0, y0, label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('mutation score')
    # plt.title('lenet5')
    # plt.grid(True)
    # plt.legend()
    #
    # # mnist
    # plt.figure(figsize=(6, 10))
    # x1 = all_rs['mnist'].values()
    # y1 = all_op_ms_by_rs['mnist'].values()
    # plt.subplot(412)
    # # plt.plot(index, lenet5_random_ratio_list, label='random')
    # plt.plot(x1, y1, label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('mutation score')
    # plt.title('mnist')
    # plt.grid(True)
    # plt.legend()
    #
    # # cifar10
    # plt.figure(figsize=(6, 10))
    # x2 = all_rs['cifar10'].values()
    # y2 = all_op_ms_by_rs['cifar10'].values()
    # plt.subplot(413)
    # # plt.plot(index, lenet5_random_ratio_list, label='random')
    # plt.plot(x2, y2, label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('mutation score')
    # plt.title('cifar10')
    # plt.grid(True)
    # plt.legend()
    #
    # # svhn
    # plt.figure(figsize=(6, 10))
    # x3 = all_rs['svhn'].values()
    # y3 = all_op_ms_by_rs['svhn'].values()
    # plt.subplot(414)
    # # plt.plot(index, lenet5_random_ratio_list, label='random')
    # plt.plot(x3, y3, label='rs')
    # plt.xlabel('mutation operator')
    # plt.ylabel('mutation score')
    # plt.title('svhn')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

