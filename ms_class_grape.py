
# 根据冗余分数对变异算子进行排序  绘制变异分数随变异算子增加的曲线
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from calculate_ms_reduced import *
from calculate_qm import sort_dict
import h5py
from calculate_ms_class import calculate_mutation_score


mutated_model_path = 'mutated_model_all'
predictions_path = 'predictions_all'
subject_list = ['lenet5', 'mnist', 'cifar10', 'svhn']

# ms_op_all = {}
# ms_op = {}
# for subject in subject_list:
#     if subject == 'lenet5':
#         op_list = ['AFRs', 'DF', 'GF', 'DM', 'LE', 'LAs', 'NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']
#     elif subject == 'mnist':
#         op_list = ['AFRs', 'DF', 'LE', 'DR', 'DM', 'LAs', 'NAI', 'NP', 'GF', 'LAa', 'WS', 'NEB', 'NS']
#     elif subject == 'cifar10':
#         op_list = ['GF', 'WS', 'NEB', 'NAI', 'NS', 'LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']
#     elif subject == 'svhn':
#         op_list = ['GF', 'NS', 'NEB', 'WS', 'AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']
#     for i in range(len(op_list)):
#         all_op_relate_class_num = {}
#         all_op_relate_killed_class_num = {}
#         for time in range(i*5, i*5+5):
#             test_adequacy_set_save_path = 'test_suit/' + 'single_class/' + subject + '/index_' + str(time) + '.h5'
#             op_relate_class_num, op_relate_killed_class_num = calculate_ms_reduced(subject, mutated_model_path, predictions_path,
#                                                                                    test_adequacy_set_save_path)
#             for op in op_relate_class_num.keys():
#                 try:
#                     all_op_relate_class_num[op] += op_relate_class_num[op]
#                     all_op_relate_killed_class_num[op] += op_relate_killed_class_num[op]
#                 except Exception as e:
#                     all_op_relate_class_num[op] = op_relate_class_num[op]
#                     all_op_relate_killed_class_num[op] = op_relate_killed_class_num[op]
#
#             try:
#                 all_op_relate_class_num['all'] += op_relate_class_num['all']
#                 all_op_relate_killed_class_num['all'] += op_relate_killed_class_num['all']
#             except Exception as e:
#                 all_op_relate_class_num['all'] = op_relate_class_num['all']
#                 all_op_relate_killed_class_num['all'] = op_relate_killed_class_num['all']
#
#         arg_ms_dic = {}
#         sum = 0
#         count = 0
#         for op in all_op_relate_class_num.keys():
#             if op != 'all':
#                 ms = round(all_op_relate_killed_class_num[op] / all_op_relate_class_num[op], 4)
#                 arg_ms_dic[op] = ms
#                 sum += ms
#                 count += 1
#         # arg_ms_dic['arg'] = round(sum / count, 2)
#         arg_ms_dic['arg'] = round(sum / count, 4)
#         arg_ms_dic['all'] = round(all_op_relate_killed_class_num['all'] / all_op_relate_class_num['all'], 4)
#         # print('###############################  ' + 'reduced' + '  ###############################################')
#         # print('arg_ms_dic:', arg_ms_dic)
#         # print('##########################################################################################')
#
#         ms_op[op_list[i]] = arg_ms_dic['all']
#         print('aaa:', ms_op)
#     ms_op_all[subject] = ms_op.copy()
#     print('bbb:', ms_op_all)
#
# print('cccccccccccbbbbbbbbbbbb:', ms_op_all)

ms_op_all = {'lenet5': {'AFRs': 0.5446, 'DF': 0.7208, 'GF': 0.8756, 'DM': 0.9163, 'LE': 0.9323, 'LAs': 0.9433, 'NAI': 0.9637, 'DR': 0.9758, 'NS': 0.9851, 'NP': 0.9895, 'NEB': 0.9945, 'WS': 0.9967, 'LAa': 1.0},
             'mnist': {'AFRs': 0.635, 'DF': 0.8051, 'GF': 0.9925, 'DM': 0.8994, 'LE': 0.8282, 'LAs': 0.9109, 'NAI': 0.9329, 'DR': 0.8652, 'NS': 1.0, 'NP': 0.9364, 'NEB': 1.0, 'WS': 0.9983, 'LAa': 0.9954},
             'cifar10': {'AFRs': 0.9973, 'DF': 1.0, 'GF': 0.6576, 'DM': 0.9906, 'LE': 0.9957, 'LAs': 0.9996, 'NAI': 0.9442, 'DR': 0.9996, 'NS': 0.9544, 'NP': 0.9906, 'NEB': 0.9189, 'WS': 0.8128, 'LAa': 0.9653, 'LR': 0.9973, 'LD': 1.0, 'LAm': 1.0},
             'svhn': {'AFRs': 0.9821, 'DF': 0.9956, 'GF': 0.7139, 'DM': 1.0, 'LE': 0.9996, 'LAs': 0.9956, 'NAI': 0.9851, 'DR': 0.9978, 'NS': 0.8535, 'NP': 1.0, 'NEB': 0.9112, 'WS': 0.9549, 'LAa': 0.9926, 'LR': 0.9973, 'LD': 1.0, 'LAm': 1.0}}
# lenet5_ms = ms_op_all['lenet5']
# mnist_ms = ms_op_all['mnist']
# cifar10_ms = ms_op_all['cifar10']
# svhn_ms = ms_op_all['svhn']

lenet5_ms = dict(sort_dict(ms_op_all['lenet5'], False))
mnist_ms = dict(sort_dict(ms_op_all['mnist'], False))
cifar10_ms = dict(sort_dict(ms_op_all['cifar10'], False))
svhn_ms = dict(sort_dict(ms_op_all['svhn'], False))

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.figure(figsize=(6, 10))
index = lenet5_ms.keys()
plt.subplot(411)
# plt.plot(index, lenet5_random_ratio_list, label='random')
plt.plot(index, lenet5_ms.values(), label='rs')
plt.xlabel('mutation operator')
plt.ylabel('mutation score')
plt.title('lenet5')
plt.grid(True)
plt.legend()

index = mnist_ms.keys()
plt.subplot(412)
# plt.plot(index, lenet5_random_num_list, label='random')
plt.plot(index, mnist_ms.values(), label='rs')
plt.xlabel('mutation operator')
plt.ylabel('mutation score')
plt.title('mnist')
plt.legend()
plt.grid(True)

index = cifar10_ms.keys()
plt.subplot(413)
# plt.plot(index, cifar10_random_ratio_list, label='random')
plt.plot(index, cifar10_ms.values(), label='rs')
plt.xlabel('mutation operator')
plt.ylabel('mutation score')
plt.title('cifar10')
plt.legend()
plt.grid(True)

index = svhn_ms.keys()
plt.subplot(414)
# plt.plot(index, lenet5_random_num_list, label='random')
plt.plot(index, svhn_ms.values(), label='rs')
plt.xlabel('mutation operator')
plt.ylabel('mutation score')
plt.title('mnist')
plt.legend()
plt.grid(True)
plt.show()