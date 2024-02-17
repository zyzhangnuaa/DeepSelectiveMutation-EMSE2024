
# 根据冗余分数对变异算子进行排序  绘制变异分数随变异算子增加的曲线
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from calculate_ms_reduced import *
from calculate_qm import sort_dict
import h5py
from calculate_ms_class import calculate_mutation_score

# ms_op_all = {'lenet5': {'NS': 45, 'DM': 58, 'NAI': 61, 'LE': 64, 'DF': 72, 'AFRs': 75, 'DR': 75, 'LAs': 75, 'WS': 76, 'NP': 77, 'NEB': 77, 'GF': 77, 'LAa': 77},
#              'mnist': {'NAI': 49, 'LE': 65, 'LAs': 67, 'DR': 70, 'AFRs': 72, 'DM': 73, 'DF': 74, 'WS': 74, 'NP': 74, 'NS': 74, 'LAa': 74, 'NEB': 74, 'GF': 74},
#              'cifar10': {'GF': 79, 'NEB': 83, 'NS': 85, 'WS': 87, 'NAI': 87, 'LAa': 87, 'LR': 88, 'AFRs': 88, 'LAs': 88, 'DR': 88, 'LE': 88, 'DM': 88, 'DF': 88, 'NP': 88},
#              'svhn': {'NS': 83, 'NEB': 86, 'GF': 87, 'WS': 87, 'NAI': 87, 'LAa': 87, 'DM': 88, 'AFRs': 88, 'LE': 88, 'DR': 88, 'NP': 88, 'DF': 88, 'LAs': 88}}

err_qs = {'lenet5': {1: 45, 2: 58, 3: 61, 4: 65, 5: 72, 6: 75, 7: 75, 8: 75, 9: 76, 10: 77, 11: 77, 12: 77, 13: 77},
             'mnist': {1: 49, 2: 65, 3: 67, 4: 70, 5: 72, 6: 73, 7: 74, 8: 74, 9: 74, 10: 74, 11: 74, 12: 74, 13: 74},
             'cifar10': {1: 79, 2: 83, 3: 85, 4: 87, 5: 87, 6: 87.5, 7: 88, 8: 88, 9: 88, 10: 88, 11: 88, 12: 88, 13: 88, 14: 88},
             'svhn': {1: 83, 2: 86, 3: 87, 4: 87, 5: 87, 6: 87, 7: 88, 8: 88, 9: 88, 10: 88, 11: 88, 12: 88, 13: 88}}

err_qs_random = {'lenet5': {1: 40.25, 2: 50.95, 3: 55.5, 4: 62, 5: 70.05, 6: 71.05, 7: 71.8, 8: 73.55, 9: 73.8, 10: 75.25, 11: 75.75, 12: 76.1, 13: 77},
             'mnist': {1: 44.7, 2: 56.35, 3: 63.50, 4: 67.65, 5: 68.85, 6: 69.6, 7: 71.0, 8: 72.15, 9: 72.55, 10: 73.15, 11: 73.35, 12: 73.8, 13: 74},
             'cifar10': {1: 71.25, 2: 78.9, 3: 83.1, 4: 83.75, 5: 85.95, 6: 86.95, 7: 87.3, 8: 87.3, 9: 87.3, 10: 87.6, 11: 87.75, 12: 87.9, 13: 88, 14: 88},
             'svhn': {1: 72.45, 2: 80.1, 3: 82.95, 4: 85.75, 5: 85.9, 6: 86.2, 7: 86.7, 8: 86.9, 9: 87.55, 10: 87.55, 11: 87.65, 12: 87.95, 13: 88}}


lenet5_qs = err_qs['lenet5']
mnist_qs = err_qs['mnist']
cifar10_qs = err_qs['cifar10']
svhn_qs = err_qs['svhn']

lenet5_qs_random = err_qs_random['lenet5']
mnist_qs_random = err_qs_random['mnist']
cifar10_qs_random = err_qs_random['cifar10']
svhn_qs_random = err_qs_random['svhn']

lenet5_qs = {key: value/77 for key, value in lenet5_qs.items()}
mnist_qs = {key: value/74 for key, value in mnist_qs.items()}
cifar10_qs = {key: value/88 for key, value in cifar10_qs.items()}
svhn_qs = {key: value/88 for key, value in svhn_qs.items()}

lenet5_qs_random = {key: value/77 for key, value in lenet5_qs_random.items()}
mnist_qs_random = {key: value/74 for key, value in mnist_qs_random.items()}
cifar10_qs_random = {key: value/88 for key, value in cifar10_qs_random.items()}
svhn_qs_random = {key: value/88 for key, value in svhn_qs_random.items()}

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# qs-model A
# plt.figure(figsize=(6, 3))
# index = lenet5_qs.keys()
# plt.subplot(111)
# plt.plot(index, lenet5_qs.values(), label='QS-based')
# plt.plot(index, lenet5_qs_random.values(), label='Random-based')
# plt.xlabel('Number of Mutation Operator')
# plt.ylabel('Fault Diversity(%)')
# # plt.title('ModelA')
# # plt.yticks(np.arange(0.55, 1.05, 0.05)) # for model A
# plt.xticks(np.arange(1, 14, 1))
# plt.grid(True)
# plt.legend(loc='lower right')
# plt.savefig('./Model_A_error_graph_qs.eps', bbox_inches='tight')
# plt.show()

# qs-model B
# plt.figure(figsize=(6, 3))
# index = mnist_qs.keys()
# plt.subplot(111)
# plt.plot(index, mnist_qs.values(), label='QS-based')
# plt.plot(index, mnist_qs_random.values(), label='Random-based')
# plt.xlabel('Number of Mutation Operator')
# plt.ylabel('Fault Diversity(%)')
# # plt.title('ModelA')
# # plt.yticks(np.arange(0.60, 1.05, 0.05)) # for model A
# plt.xticks(np.arange(1, 14, 1))
# plt.grid(True)
# plt.legend(loc='lower right')
# plt.savefig('./Model_B_error_graph_qs.eps', bbox_inches='tight')
# plt.show()

# qs-model C
# plt.figure(figsize=(6, 3))
# index = svhn_qs.keys()
# plt.subplot(111)
# plt.plot(index, svhn_qs.values(), label='QS-based')
# plt.plot(index, svhn_qs_random.values(), label='Random-based')
# plt.xlabel('Number of Mutation Operator')
# plt.ylabel('Fault Diversity(%)')
# # plt.title('ModelA')
# # plt.yticks(np.arange(0.80, 1.05, 0.05)) # for model A
# plt.xticks(np.arange(1, 14, 1))
# plt.grid(True)
# plt.legend(loc='lower right')
# plt.savefig('./Model_C_error_graph_qs.eps', bbox_inches='tight')
# plt.show()

# qs-model D
plt.figure(figsize=(6, 3))
index = cifar10_qs.keys()
plt.subplot(111)
plt.plot(index, cifar10_qs.values(), label='QS-based')
plt.plot(index, cifar10_qs_random.values(), label='Random-based')
plt.xlabel('Number of Mutation Operator')
plt.ylabel('Fault Diversity(%)')
# plt.title('ModelA')
plt.yticks(np.arange(0.80, 1.025, 0.05)) # for model A
plt.xticks(np.arange(1, 14, 1))
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('./Model_D_error_graph_qs.eps', bbox_inches='tight')
plt.show()

