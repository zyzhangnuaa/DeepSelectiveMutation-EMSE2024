
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

# ms_op_all = {'lenet5': {'NS': 45, 'DM': 58, 'NAI': 61, 'LE': 64, 'DF': 72, 'AFRs': 75, 'DR': 75, 'LAs': 75, 'WS': 76, 'NP': 77, 'NEB': 77, 'GF': 77, 'LAa': 77},
#              'mnist': {'NAI': 49, 'LE': 65, 'LAs': 67, 'DR': 70, 'AFRs': 72, 'DM': 73, 'DF': 74, 'WS': 74, 'NP': 74, 'NS': 74, 'LAa': 74, 'NEB': 74, 'GF': 74},
#              'cifar10': {'GF': 79, 'NEB': 83, 'NS': 85, 'WS': 87, 'NAI': 87, 'LAa': 87, 'LR': 88, 'AFRs': 88, 'LAs': 88, 'DR': 88, 'LE': 88, 'DM': 88, 'DF': 88, 'NP': 88},
#              'svhn': {'NS': 83, 'NEB': 86, 'GF': 87, 'WS': 87, 'NAI': 87, 'LAa': 87, 'DM': 88, 'AFRs': 88, 'LE': 88, 'DR': 88, 'NP': 88, 'DF': 88, 'LAs': 88}}

ms_op_all = {'lenet5': {1: 45, 2: 58, 3: 61, 4: 64, 5: 72, 6: 75, 7: 75, 8: 75, 9: 76, 10: 77, 11: 77, 12: 77, 13: 77},
             'mnist': {1: 49, 2: 65, 3: 67, 4: 70, 5: 72, 6: 73, 7: 74, 8: 74, 9: 74, 10: 74, 11: 74, 12: 74, 13: 74},
             'cifar10': {1: 79, 2: 83, 3: 85, 4: 87, 5: 87, 6: 87, 7: 88, 8: 88, 9: 88, 10: 88, 11: 88, 12: 88, 13: 88, 14: 88},
             'svhn': {1: 83, 2: 86, 3: 87, 4: 87, 5: 87, 6: 87, 7: 88, 8: 88, 9: 88, 10: 88, 11: 88, 12: 88, 13: 88}}


lenet5_ms = ms_op_all['lenet5']
mnist_ms = ms_op_all['mnist']
cifar10_ms = ms_op_all['cifar10']
svhn_ms = ms_op_all['svhn']

lenet5_ms = {key: value/77 for key, value in lenet5_ms.items()}
mnist_ms = {key: value/74 for key, value in mnist_ms.items()}
cifar10_ms = {key: value/88 for key, value in cifar10_ms.items()}
svhn_ms = {key: value/88 for key, value in svhn_ms.items()}

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.figure(figsize=(6, 10))
index = lenet5_ms.keys()
plt.subplot(411)
# plt.plot(index, lenet5_random_ratio_list, label='random')
plt.plot(index, lenet5_ms.values(), label='rs')
plt.xlabel('Number of mutation operator')
plt.ylabel('Label Diversity(%)')
plt.title('ModelA')
plt.xticks(np.arange(1, 14, 1))
plt.grid(True)
plt.legend()

index = mnist_ms.keys()
plt.subplot(412)
# plt.plot(index, lenet5_random_num_list, label='random')
plt.plot(index, mnist_ms.values(), label='rs')
plt.xlabel('Number of mutation operator')
plt.ylabel('Label Diversity(%)')
plt.title('ModelB')
plt.xticks(np.arange(1, 14, 1))
plt.legend()
plt.grid(True)

index = svhn_ms.keys()
plt.subplot(413)
# plt.plot(index, lenet5_random_num_list, label='random')
plt.plot(index, svhn_ms.values(), label='rs')
plt.xlabel('Number of mutation operator')
plt.ylabel('Label Diversity(%)')
plt.title('ModelC')
plt.xticks(np.arange(1, 14, 1))
plt.legend()
plt.grid(True)

index = cifar10_ms.keys()
plt.subplot(414)
# plt.plot(index, cifar10_random_ratio_list, label='random')
plt.plot(index, cifar10_ms.values(), label='rs')
plt.xlabel('mutation operator')
plt.ylabel('Label Diversity(%)')
plt.title('ModelD')
plt.xticks(np.arange(1, 16, 1))
plt.legend()
plt.grid(True)
plt.show()
