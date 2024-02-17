import csv
import os
import glob
from minimal_test_suit import minimal_test_suit_index_reduce
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


# 使用划分好的变异算子构建的测试充分集计算不同冗余分数下变异算子的变异分数
def op_Ms_By_Rs(mutated_model_path, predictions_path, subject_name, count):  # count表示根据冗余分数划分的段数
    test_kill_class = print_test_kill_class(mutated_model_path, predictions_path, subject_name)
    print('op_test_kill_class_num:', test_kill_class)
    unreduntant_class = print_unreduntant_class(mutated_model_path, predictions_path, subject_name)
    print('op_unreduntant_class_num:', unreduntant_class)

    op_ms_by_rs = {}
    for i in range(count):
        all_op_relate_class_num = {}
        all_op_relate_killed_class_num = {}
        for time in range(i * 5, i * 5 + 5):
            test_adequacy_set_save_path = 'test_suit/' + 'seg_class_rs/' + subject_name + '/index_' + str(time) + '.h5'
            op_relate_class_num, op_relate_killed_class_num = calculate_ms_reduced(subject_name, mutated_model_path,
                                                                                   predictions_path,
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
        op_ms_by_rs[i] = arg_ms_dic['all']

    return op_ms_by_rs


if __name__ == '__main__':
    subject_name = 'lenet5'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    # 根据冗余分数划分变异算子，并构建最小的测试用例集
    # mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    # op_set = set()
    # for mutant in mutants_path:
    #     op_name = (mutant.split("\\"))[-1].replace(".h5", "").split('_')[1]
    #     op_set.add(op_name)
    # op_list = list(op_set)
    # print(op_list)
    # print(len(op_list))
    #
    # op_rs_path = os.path.join(predictions_path, subject_name, "rs.csv")
    # op_rs =read_set_from_csv(op_rs_path)
    # print(op_rs)
    #
    # op_list_choiced = []
    # rs = 0
    # for item in op_rs.items():
    #     rs = item[1]
    #     print('minimal_rs:', rs)
    #     break
    # num = 0
    # count = 1
    # all_rs = {}  # 记录所有不同的冗余分数
    # all_rs[0] = rs
    # while set(op_list_choiced) != set(op_list):
    #     op_list_choiced = []
    #     for item in op_rs.items():
    #         if item[1] <= rs:
    #             op_list_choiced.append(item[0][1:])
    #         else:
    #             rs = item[1]
    #             all_rs[count] = rs
    #             count += 1
    #             break
    #     print('aa:', op_list_choiced)
    #     reduce_op = list(set(op_list) - set(op_list_choiced))
    #     print('lllllllllllllll:', reduce_op)
    #
    #     # 使用选择的变异算子构建最小测试用例集
    #     # for n in range(num, num+5):
    #     #     kind = 'seg_class_rs'
    #     #     minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, n, reduce_op)
    #     # num += 5
    # print('count:', count)  # 不同的冗余分数的数量 也就是划分的段数
    # print('all_rs:', all_rs)

    # 下面的代码用于根据冗余分数对变异算子进行划分，计算变异分数，并绘制折线图
    # count: lenet5-11 mnist-10 cifar10-9 svhn-8
    # all_rs
    # 根据冗余分数对变异算子进行划分 计算得到的变异分数
    count = {'lenet5': 11, 'mnist': 10, 'cifar10': 9, 'svhn': 8}
    all_rs = {'lenet5': {0: 0.8613, 1: 0.8686, 2: 0.8978, 3: 0.9051, 4: 0.9197, 5: 0.9343,
                         6: 0.9416, 7: 0.9489, 8: 0.9635, 9: 0.9708, 10: 0.9927},
              'mnist': {0: 0.7823, 1: 0.8468, 2: 0.8548, 3: 0.9194, 4: 0.9274, 5: 0.9435,
                        6: 0.9597, 7: 0.9677, 8: 0.9919, 9: 1.0},
              'cifar10': {0: 0.9217, 1: 0.9452, 2: 0.9478, 3: 0.9634, 4: 0.9713, 5: 0.9869,
                          6: 0.9922, 7: 0.9974, 8: 1.0},
              'svhn': {0: 0.93, 1: 0.9481, 2: 0.9639, 3: 0.9661, 4: 0.991, 5: 0.9955,
                       6: 0.9977, 7: 1.0}}
    # 可以直接使用之前计算的
    # all_op_ms_by_rs = {}
    # for key in count.keys():
    #     op_ms_by_rs = op_Ms_By_Rs(mutated_model_path, predictions_path, key, count[key])
    #     print('subject_name:', key)
    #     print('count:', count[key])  # 根据冗余分数划分的段数
    #     print('aaaaaaaa:', all_rs[key])  # 所有不同的冗余分数
    #     print(op_ms_by_rs)  # 根据冗余分数划分变异算子计算得到的变异分数
    #     all_op_ms_by_rs[key] = op_ms_by_rs

    all_op_ms_by_rs = {'lenet5': {0: 0.5446, 1: 0.7208, 2: 0.8756, 3: 0.9163, 4: 0.9323, 5: 0.9637,
                                  6: 0.9758, 7: 0.9851, 8: 0.9895, 9: 0.9967, 10: 1.0},
                       'mnist': {0: 0.635, 1: 0.8051, 2: 0.8282, 3: 0.8994, 4: 0.9109, 5: 0.9329,
                                 6: 0.9925, 7: 0.9983, 8: 1.0, 9: 1.0},
                       'cifar10': {0: 0.6576, 1: 0.8128, 2: 0.9189, 3: 0.9442, 4: 0.9544, 5: 0.9653,
                                   6: 0.9902, 7: 0.9953, 8: 1.0},
                       'svhn': {0: 0.7139, 1: 0.8535, 2: 0.9112, 3: 0.9549, 4: 0.9926, 5: 0.9956,
                                6: 0.9991, 7: 1.0}}
    print('all_rs:', all_rs)
    print('all_op_ms_by_rs:', all_op_ms_by_rs)

    # 根据冗余分数对应的变异分数，绘制折线图
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    plt.figure(figsize=(6, 10))
    index = all_rs['lenet5'].values()
    plt.subplot(411)
    # plt.plot(index, lenet5_random_ratio_list, label='random')
    plt.plot(index, all_op_ms_by_rs['lenet5'].values(), label='rs')
    plt.xlabel('mutation operator')
    plt.ylabel('mutation score')
    plt.title('lenet5')
    plt.grid(True)
    plt.legend()

    index = all_rs['mnist'].values()
    plt.subplot(412)
    # plt.plot(index, lenet5_random_num_list, label='random')
    plt.plot(index, all_op_ms_by_rs['mnist'].values(), label='rs')
    plt.xlabel('mutation operator')
    plt.ylabel('mutation score')
    plt.title('mnist')
    plt.legend()
    plt.grid(True)

    index = all_rs['cifar10'].values()
    plt.subplot(413)
    # plt.plot(index, cifar10_random_ratio_list, label='random')
    plt.plot(index, all_op_ms_by_rs['cifar10'].values(), label='rs')
    plt.xlabel('mutation operator')
    plt.ylabel('mutation score')
    plt.title('cifar10')
    plt.legend()
    plt.grid(True)

    index = all_rs['svhn'].values()
    plt.subplot(414)
    # plt.plot(index, lenet5_random_num_list, label='random')
    plt.plot(index, all_op_ms_by_rs['svhn'].values(), label='rs')
    plt.xlabel('mutation operator')
    plt.ylabel('mutation score')
    plt.title('svhn')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # lenet5
    # plt.figure(figsize=(6, 10))
    # x0 = all_rs['lenet5'].values()
    # y0 = all_op_ms_by_rs['lenet5'].values()
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

