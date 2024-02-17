
# 根据冗余分数对变异算子进行排序  绘制变异分数随变异算子增加的曲线
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from calculate_ms_reduced import *
from calculate_qm import sort_dict
import h5py
from calculate_ms_class import calculate_mutation_score

err_rs = {'lenet5': {1: 37, 2: 50, 3: 56, 4: 63, 5: 67, 6: 73, 7: 75, 8: 76, 9: 76, 10: 77, 11: 77, 12: 77, 13: 77},
             'mnist': {1: 48, 2: 58, 3: 63, 4: 65, 5: 70, 6: 71, 7: 73, 8: 73, 9: 74, 10: 74, 11: 74, 12: 74, 13: 74},
             'cifar10': {1: 65, 2: 79, 3: 81, 4: 87, 5: 87, 6: 87, 7: 88, 8: 88, 9: 89, 10: 89, 11: 89, 12: 89, 13: 89, 14: 89},
             'svhn': {1: 72, 2: 83, 3: 86, 4: 87, 5: 88, 6: 88, 7: 88, 8: 88, 9: 88, 10: 88, 11: 88, 12: 88, 13: 88}}

err_rs_random = {'lenet5': {1: 19.7, 2: 27.35, 3: 41.5, 4: 46.9, 5: 51.95, 6: 57, 7: 64.3, 8: 64.65, 9: 69, 10: 71.4, 11: 74.25, 12: 75.8, 13: 77},
             'mnist': {1: 17.15, 2: 27.45, 3: 39.95, 4: 44.4, 5: 50.15, 6: 53.2, 7: 58.4, 8: 62.1, 9: 64.7, 10: 68.2, 11: 68.95, 12: 72.85, 13: 74},
             'cifar10': {1: 35.6, 2: 61, 3: 64.3, 4: 69.65, 5: 74, 6: 79.55, 7: 80.75, 8: 83.15, 9: 83.9, 10: 84.95, 11: 84.9, 12: 88.1, 13: 89, 14: 89},
             'svhn': {1: 47.55, 2: 62.2, 3: 67.7, 4: 76.7, 5: 80.5, 6: 81.75, 7: 84.65, 8: 85, 9: 86.8, 10: 87, 11: 86.9, 12: 88, 13: 88}}

lenet5_rs = err_rs['lenet5']
mnist_rs = err_rs['mnist']
cifar10_rs = err_rs['cifar10']
svhn_rs = err_rs['svhn']

lenet5_rs_random = err_rs_random['lenet5']
mnist_rs_random = err_rs_random['mnist']
cifar10_rs_random = err_rs_random['cifar10']
svhn_rs_random = err_rs_random['svhn']

lenet5_rs = {key: value/77 for key, value in lenet5_rs.items()}
mnist_rs = {key: value/74 for key, value in mnist_rs.items()}
cifar10_rs = {key: value/89 for key, value in cifar10_rs.items()}
svhn_rs = {key: value/88 for key, value in svhn_rs.items()}

lenet5_rs_random = {key: value/77 for key, value in lenet5_rs_random.items()}
mnist_rs_random = {key: value/74 for key, value in mnist_rs_random.items()}
cifar10_rs_random = {key: value/89 for key, value in cifar10_rs_random.items()}
svhn_rs_random = {key: value/88 for key, value in svhn_rs_random.items()}

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# qs-model A
plt.figure(figsize=(6, 3))
index = lenet5_rs.keys()
plt.subplot(111)
plt.plot(index, lenet5_rs.values(), label='RS-based')
plt.plot(index, lenet5_rs_random.values(), label='Random-based')
plt.xlabel('Number of Mutation Operator')
plt.ylabel('Fault Diversity(%)')
# plt.title('ModelA')
plt.yticks(np.arange(0.20, 1.05, 0.1)) # for model A
plt.xticks(np.arange(1, 14, 1))
plt.grid(True)
plt.legend()
plt.savefig('./Model_A_error_graph_rs.eps', bbox_inches='tight')
# plt.savefig('./Model_A_error_graph_rs.png', bbox_inches='tight')
plt.show()

# qs-model B
# plt.figure(figsize=(6, 3))
# index = mnist_rs.keys()
# plt.subplot(111)
# plt.plot(index, mnist_rs.values(), label='RS-based')
# plt.plot(index, mnist_rs_random.values(), label='Random-based')
# plt.xlabel('Number of Mutation Operator')
# plt.ylabel('Fault Diversity(%)')
# # plt.title('ModelA')
# plt.yticks(np.arange(0.20, 1.05, 0.10)) # for model A
# plt.xticks(np.arange(1, 14, 1))
# plt.grid(True)
# plt.legend()
# plt.savefig('./Model_B_error_graph_rs.eps', bbox_inches='tight')
# plt.show()

# qs-model C
# plt.figure(figsize=(6, 3))
# index = svhn_rs.keys()
# plt.subplot(111)
# plt.plot(index, svhn_rs.values(), label='RS-based')
# plt.plot(index, svhn_rs_random.values(), label='Random-based')
# plt.xlabel('Number of Mutation Operator')
# plt.ylabel('Fault Diversity(%)')
# # plt.title('ModelA')
# # plt.yticks(np.arange(0.80, 1.05, 0.05)) # for model A
# plt.xticks(np.arange(1, 14, 1))
# plt.grid(True)
# plt.legend()
# plt.savefig('./Model_C_error_graph_rs.eps', bbox_inches='tight')
# plt.show()

# qs-model D
plt.figure(figsize=(6, 3))
index = cifar10_rs.keys()
plt.subplot(111)
plt.plot(index, cifar10_rs.values(), label='RS-based')
plt.plot(index, cifar10_rs_random.values(), label='Random-based')
plt.xlabel('Number of Mutation Operator')
plt.ylabel('Fault Diversity(%)')
# plt.title('ModelA')
# plt.yticks(np.arange(0.80, 1.025, 0.1)) # for model A
plt.xticks(np.arange(1, 14, 1))
plt.grid(True)
plt.legend()
plt.savefig('./Model_D_error_graph_rs.eps', bbox_inches='tight')
plt.show()


