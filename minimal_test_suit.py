import random
import time
import h5py
import pulp
import pandas as pd
from calculate_ms import read_set_from_csv
from kill_all_related_class import related_class
import os
from keras.datasets import mnist, cifar10
import numpy as np


def minimal_test_suit(subject_name, mutated_model_path, predictions_path, kind, num):

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    test_set = set()  # 记录所有杀死的测试输入
    for mutant in related_class_dict:
        if len(related_class_dict[mutant]) == 0:
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
            test_set.update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    print(all_mutant_info)
    print(test_set)
    print(len(test_set))

    start_time = time.clock()
    # 构建最小的测试充分集
    prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)

    # 目标函数
    prob += pulp.lpSum([x[i] for i in x])

    # 约束函数
    for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
        class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
        kill_class = class_input_info.keys()
        for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
            kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
            el = {t: 0 for t in test_set}
            for t in kill_input:
                el[t] = 1
            # 定义约束函数
            prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1

    prob.solve()

    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)

    select_input = []
    for v in prob.variables():
        print(v.name, '=', v.varValue)
        if v.varValue == 1:
            select_input.append(int(v.name.split('_')[-1]))
    print('最小测试充分集的大小为:', pulp.value(prob.objective))
    print('select_input:', select_input)

    if subject_name == 'mnist' or subject_name == 'lenet5':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_test = x_test.astype('float32')
        x_test /= 255
    test_x = list()
    test_y = list()
    for ind in select_input:
        test_x.append(x_test[ind])
        test_y.append(y_test[ind])

    if not os.path.exists(os.path.join('test_suit', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'data_' + str(num) + '.h5'), 'w')
    hf.create_dataset('x_test', data=test_x)
    hf.create_dataset('y_test', data=test_y)
    hf.close()


def minimal_test_suit_index(subject_name, mutated_model_path, predictions_path, kind, num, op):  # 保存所选测试输入的索引
    # 为除去变异算子op的变异体构建最小的测试充分集

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")

    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    test_set = set()  # 记录所有杀死的测试输入
    for mutant in related_class_dict:
        if mutant.split('_')[1] == op:  # 不考虑变异算子op的变异体
            continue
        if len(related_class_dict[mutant]) == 0:
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
            test_set.update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    print(all_mutant_info)
    print(test_set)
    print(len(test_set))

    start_time = time.clock()
    # 构建最小的测试充分集
    prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)

    # 目标函数
    prob += pulp.lpSum([x[i] for i in x])

    # 约束函数
    for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
        class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
        kill_class = class_input_info.keys()
        for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
            kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
            el = {t: 0 for t in test_set}
            for t in kill_input:
                el[t] = 1
            # 定义约束函数
            prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1

    prob.solve()

    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)

    select_input = []
    for v in prob.variables():
        print(v.name, '=', v.varValue)
        if v.varValue == 1:
            select_input.append(int(v.name.split('_')[-1]))
    print('最小测试充分集的大小为:', pulp.value(prob.objective))
    print('select_input:', select_input)

    if not os.path.exists(os.path.join('test_suit', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'index_' + str(num) + '.h5'), 'w')
    hf.create_dataset('index', data=select_input)
    hf.close()


def minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, op_list):
    # 为约减后的变异算子构建最小的测试充分集 11
    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    test_set = set()  # 记录所有杀死的测试输入
    mutant_num = 0
    for mutant in related_class_dict:
        if mutant.split('_')[1] in op_list:  # 不考虑变异算子op的变异体
            continue
        mutant_num += 1
        if len(related_class_dict[mutant]) == 0:
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
            test_set.update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    print(all_mutant_info)
    print(test_set)
    print(len(test_set))
    print('约减后的变异体数量为:', mutant_num)

    start_time = time.clock()
    # 构建最小的测试充分集
    prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)
    print("aaaa:", x)
    for i in x:
        print(i)
        break

    # 目标函数
    prob += pulp.lpSum([x[i] for i in x])

    # 约束函数
    for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
        class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
        kill_class = class_input_info.keys()
        for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
            kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
            el = {t: 0 for t in test_set}
            for t in kill_input:
                el[t] = 1
            # 定义约束函数
            prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1

    prob.solve()

    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)

    select_input = []
    for v in prob.variables():
        print(v.name, '=', v.varValue)
        if v.varValue == 1:
            select_input.append(int(v.name.split('_')[-1]))
    print('最小测试充分集的大小为:', pulp.value(prob.objective))
    print('select_input:', select_input)

    if not os.path.exists(os.path.join('test_suit_newexp', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit_newexp', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit_newexp', kind, subject_name, 'index_' + str(num) + '.h5'), 'w')
    hf.create_dataset('index', data=select_input)
    hf.close()


def minimal_test_suit_index_reduce_single_class(subject_name, mutated_model_path, predictions_path, kind, num, op_list):
    # 为约减后的变异算子构建最小的测试充分集
    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    test_set = set()  # 记录所有杀死的测试输入
    mutant_num = 0
    for mutant in related_class_dict:
        if mutant.split('_')[1] in op_list:  # 不考虑变异算子op的变异体
            continue
        mutant_num += 1
        if len(related_class_dict[mutant]) == 0:
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
            test_set.update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    print(all_mutant_info)
    print(test_set)
    print(len(test_set))
    print('约减后的变异体数量为:', mutant_num)

    start_time = time.clock()
    # 构建最小的测试充分集
    prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)

    # 目标函数
    prob += pulp.lpSum([x[i] for i in x])

    # 约束函数
    for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
        class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
        kill_class = class_input_info.keys()
        for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
            kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
            el = {t: 0 for t in test_set}
            for t in kill_input:
                el[t] = 1
            # 定义约束函数
            prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1

    prob.solve()

    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)

    select_input = []
    for v in prob.variables():
        print(v.name, '=', v.varValue)
        if v.varValue == 1:
            select_input.append(int(v.name.split('_')[-1]))
    print('最小测试充分集的大小为:', pulp.value(prob.objective))
    print('select_input:', select_input)

    if not os.path.exists(os.path.join('test_suit', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'index_' + str(num) + '.h5'), 'w')
    hf.create_dataset('index', data=select_input)
    hf.close()


def minimal_test_suit_op_class(subject_name, mutated_model_path, predictions_path, kind, num, op):

    # # 加载测试充分集
    # test_adequacy_set_save_path = os.path.join('test_suit', 'all_class', subject_name, 'index_0.h5')
    # hf = h5py.File(test_adequacy_set_save_path, 'r')
    # index = np.asarray(hf.get('index'))

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")  不用
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    test_set_class = {}
    for i in range(10):
        test_set_class[i] = set()
    # test_set = set()  # 记录所有杀死的测试输入
    for mutant in related_class_dict:
        if mutant.split('_')[1] != op:
            continue
        if len(related_class_dict[mutant]) == 0:  # 变异体上没有可以被杀死的类别
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)])))
            test_set_class[int(c)].update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    print(all_mutant_info)
    print(test_set_class)

    select_input_all_class_dict = {}
    start_time = time.clock()
    # 为变异算子op产生的变异体在每个类别上的变异体类别对构建最小的测试充分集
    for i in range(10):
        if len(test_set_class[i]) == 0:
            select_input_all_class_dict[i] = []
            continue

        prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
        x = pulp.LpVariable.dicts('x', list(test_set_class[i]), lowBound=0, upBound=1, cat=pulp.LpInteger)
        # 目标函数
        prob += pulp.lpSum([x[i] for i in x])
        # 约束函数
        for mutant in all_mutant_info.keys():
            try:
                # 对每个变异体上的每个可杀死的类都要有一个约束函数， 保证其至少被一个测试输入杀死
                class_input_info = all_mutant_info[mutant][i]  # 变异体在类i上的杀死输入
                el = {t: 0 for t in test_set_class[i]}
                for t in class_input_info:
                    el[t] = 1
                # 定义约束函数
                prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1
            except Exception as e:  #  变异体上的这个类不能被杀死
                continue

        prob.solve()

        select_input = []
        for v in prob.variables():
            print(v.name, '=', v.varValue)
            if v.varValue == 1:
                select_input.append(int(v.name.split('_')[-1]))
        print('最小测试充分集的大小为:', pulp.value(prob.objective))
        print('select_input_of_class_' + str(i) + ':', select_input)

        select_input_all_class_dict[i] = select_input

    print(select_input_all_class_dict)

    # 记录选择的测试输入的索引
    if not os.path.exists(os.path.join('test_suit', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'index_' + str(num) + '.h5'), 'w')
    for i in range(10):
        hf.create_dataset('index_' + str(i), data=select_input_all_class_dict[i])
    hf.close()


def minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, num, op_list):
    # 为约减后的变异算子构建最小的测试充分集

    # 加载测试充分集
    test_adequacy_set_save_path = os.path.join('test_suit', 'all_class', subject_name, 'index_0.h5')
    hf = h5py.File(test_adequacy_set_save_path, 'r')
    index = np.asarray(hf.get('index'))

    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)

    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    all_mutant_info = {}
    # test_set = set()  # 记录所有杀死的测试输入
    mutant_num = 0
    for mutant in related_class_dict:
        if mutant.split('_')[1] in op_list:  # 不考虑变异算子op的变异体
            continue
        mutant_num += 1
        if len(related_class_dict[mutant]) == 0:
            continue
        mutant_info = {}
        kill_class = related_class_dict[mutant]
        for c in kill_class:
            # 获取杀死该类别的输入
            kill_input = set(list(eval(all_mutant_class_input_killing_info[mutant][int(c)]))).intersection(set(index))
            # test_set.update(kill_input)
            mutant_info[int(c)] = kill_input
        all_mutant_info[mutant] = mutant_info
    # print(all_mutant_info)
    # print(test_set)
    # print(len(test_set))
    # print(index)
    # print(len(index))
    print('约减后的变异体数量为:', mutant_num)

    start_time = time.clock()
    # 构建最小的测试充分集
    prob = pulp.LpProblem(name='minimal_test_suit', sense=pulp.LpMinimize)
    # x = pulp.LpVariable.dicts('x', list(test_set), lowBound=0, upBound=1, cat=pulp.LpInteger)
    x = pulp.LpVariable.dicts('x', list(index), lowBound=0, upBound=1, cat=pulp.LpInteger)

    # 目标函数
    prob += pulp.lpSum([x[i] for i in x])

    # 约束函数
    for mutant in all_mutant_info.keys():  # {'lenet5_AFRs_0.01_mutated_1': {9: {813, 2129, 2582, 6166, 4761, 9692}}, 'lenet5_AFRs_0.01_mutated_10': {0: {8325}, 3: {2953}}
        class_input_info = all_mutant_info[mutant]  # {9: {813, 2129, 2582, 6166, 4761, 9692}}
        kill_class = class_input_info.keys()
        for c in kill_class:  # 对每个变异体上的每个类都要有一个约束函数， 保证其至少被一个测试输入杀死
            kill_input = class_input_info[c]  # for 9, kill_input = {813, 2129, 2582, 6166, 4761, 9692}
            # el = {t: 0 for t in test_set}
            el = {t: 0 for t in index}
            for t in kill_input:
                el[t] = 1
            # 定义约束函数
            prob += pulp.lpSum([el[i] * x[i] for i in x]) >= 1

    prob.solve()

    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)

    select_input = []
    for v in prob.variables():
        # print(v.name, '=', v.varValue)
        if v.varValue == 1:
            select_input.append(int(v.name.split('_')[-1]))
    print('最小测试充分集的大小为:', pulp.value(prob.objective))
    print('select_input:', select_input)
    if subject_name == 'lenet5':
        all = 123
    elif subject_name == 'mnist':
        all = 119
    elif subject_name == 'cifar10':
        all = 93
    elif subject_name == 'svhn':
        all = 95
    print('test lost:', round((all-pulp.value(prob.objective))/all * 100, 2))

    if not os.path.exists(os.path.join('test_suit_newexp', kind, subject_name)):
        try:
            os.makedirs(os.path.join('test_suit_newexp', kind, subject_name))
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    hf = h5py.File(os.path.join('test_suit_newexp', kind, subject_name, 'index_' + str(num) + '.h5'), 'w')
    hf.create_dataset('index', data=select_input)
    hf.close()


if __name__ == '__main__':

    subject_name = 'cifar10'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    # 1、为所有的变异算子构建最小的测试充分集   在所有类别上构建
    # for num in range(1):
    #     kind = 'all_class'
    #     minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, [])

    # 2、为除去变异算子op的所有变异体构建测试充分集  用于计算冗余分数  在非冗余的类别上进行构建
    # if subject_name == 'lenet5' or subject_name == 'mnist' or subject_name == 'svhn':
    #     op_list = ['WGF', 'WNAI', 'WNEB', 'WNS', 'WWS', 'WLAa', 'WDR', 'WLE', 'WDM', 'WDF', 'WNP', 'WLAs', 'WAFRs']
    # else:
    #     op_list = ['WGF', 'WNAI', 'WNEB', 'WNS', 'WWS', 'WLAa', 'WDR', 'WLE', 'WDM', 'WDF', 'WNP', 'WLAs', 'WAFRs',
    #                'WLR', 'WLD', 'WLAm']
    # for num in range(5):
    #     for op in op_list:
    #         kind = op + '_class'
    #         print('cccbbb:', op[1:])
    #         minimal_test_suit_index(subject_name, mutated_model_path, predictions_path, kind, num, op[1:])

    # 3、为某个变异算子产生的变异体上的每个类别构建测试充分集, 保存在一个index_i.h5  用于计算质量分数  所有可杀死类别上构建
    # if subject_name == 'lenet5' or subject_name == 'mnist' or subject_name == 'svhn':
    #     op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']
    # else:
    #     op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs', 'LR', 'LD', 'LAm']
    # for num in range(5):
    #     for op in op_list:
    #         kind = op + '_class'
    #         minimal_test_suit_op_class(subject_name, mutated_model_path, predictions_path, kind, num, op)

    # 4、为约减后的变异算子构建测试充分集       在所有可杀死的类上进行构建
    # for num in range(5):
    #     kind = 'reduced_class_rs_and_qs'
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'DR', 'NP', 'LAs']  # lenet5    reduced_class_1   0-4 非冗余类 5-9所有类
    #     # reduce_op = ['GF', 'NEB', 'NS', 'WS', 'LAa', 'NP']  # mnist
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'LAs', 'AFRs', 'LD', 'LAm']  # cifar10
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # svhn
    #
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'DR', 'NP', 'NS', 'GF']  # lenet5
    #     # reduce_op = ['GF', 'NEB', 'NS', 'WS', 'LAa', 'NP']  # mnist
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'LAs', 'AFRs', 'LD', 'LAm', 'NP']  # cifar10
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # svhn
    #
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'GF', 'NP']  # lenet5
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'GF', 'NS']  # mnist
    #     # reduce_op = ['LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm', 'NP']  # cifar10
    #     # reduce_op = ['DR', 'DM', 'LE', 'NP', 'DF', 'LAs']  # svhn
    #
    #     # reduce_op = ['GF', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'NP', 'LAs']  # lenet5
    #     # reduce_op = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'NP']  # mnist
    #     # reduce_op = ['NAI', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # svhn
    #     reduce_op = ['LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs', 'LR', 'LD', 'LAm']  # cifar10
    #     #
    #     minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)

    # 约减后的变异算子的测试损失  在所有可杀死的类别上
    # for num in range(5):
    #     kind = 'reduced_class_test_loss'
    #
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'DR', 'NP', 'LAs']  # lenet5    reduced_class_1   0-4 非冗余类 5-9所有类
    #     # reduce_op = ['GF', 'NEB', 'NS', 'WS', 'LAa', 'NP']  # mnist
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'LAs', 'AFRs', 'LD', 'LAm']  # cifar10
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # svhn
    #
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'DR', 'NP', 'NS', 'GF']  # lenet5
    #     # reduce_op = ['GF', 'NEB', 'NS', 'WS', 'LAa', 'NP']  # mnist
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'LAs', 'AFRs', 'LD', 'LAm', 'NP']  # cifar10
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']  # svhn
    #
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'GF', 'NP']  # lenet5
    #     # reduce_op = ['NEB', 'WS', 'LAa', 'GF', 'NS']  # mnist
    #     # reduce_op = ['LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm', 'NP']  # cifar10
    #     reduce_op = ['DR', 'DM', 'LE', 'NP', 'DF', 'LAs']  # svhn
    #
    #     minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)


    # RS对比实验--为约减后的变异算子构建测试充分集  对比随机  冗余分数   在非冗余的类上进行构建
    for num in range(240, 260):
        # kind = 'random_class'
        # kind = 'random_class_rs'
        kind = 'random_seg_class_rs'
        if subject_name == 'lenet5' or subject_name == 'mnist' or subject_name == 'svhn':
            op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']
        else:
            op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs', 'LR']
        # lenet5:10,9,6,1  mnist：11,7,4,4  cifar10: 12,11,9,7  svhn: 11,10,9,6
        reduce_op = random.sample(op_list, 1)  # lenet5: 10,9,6,4,2  mnist:11,7,4,4,4  cifar10:12,11,9,7  svhn:11,10,9,8,6  past
        reduce_op.append('LD')
        reduce_op.append('LAm')
        print('choice_op:', set(op_list) - set(reduce_op))
        print('reduce_op:', reduce_op)
        # reduce_op = ['WS', 'GF', 'NAI', 'NEB', 'LAa', 'NS']  # lenet5
        # reduce_op = ['GF', 'NEB', 'NS', 'LAa', 'WS', 'NAI']  # mnist
        # reduce_op = ['DR', 'LE', 'DM', 'DF', 'LAs', 'LD', 'LAm', 'AFRs', 'NP']  # cifar10
        # reduce_op = ['DM', 'DF', 'DR', 'LE', 'LAs', 'NP']  # svhn

        minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)

    # QS对比实验-为约减后的变异算子构建测试充分集   对比随机  质量分数   在所有类上进行构建
    # for num in range(240,260):
    #     kind = 'random_seg_class_qs'
    #     if subject_name == 'lenet5' or subject_name == 'mnist' or subject_name == 'svhn':
    #         op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs']
    #     else:
    #         op_list = ['GF', 'NAI', 'NEB', 'NS', 'WS', 'LAa', 'DR', 'LE', 'DM', 'DF', 'NP', 'LAs', 'AFRs', 'LR']
    #     # lenet5:7,5,3,2  mnist：8,5,4,2  cifar10: 10,-,7,5  svhn: 10,9,-,3
    #     reduce_op = random.sample(op_list, 1)
    #     reduce_op.append('LD')
    #     reduce_op.append('LAm')
    #     print('choice_op:', set(op_list) - set(reduce_op))
    #     print('reduce_op:', reduce_op)
    #
    #     minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)

    # 分段构建测试充分集 根据冗余分数分组 在非冗余的类别上构建 增加实验部分
    # for num in range(20,25):
    #     kind = 'seg_class_rs'
    #     # # for lenet5
    #     # before
    #     # reduce_op = ['DM', 'LE', 'LAs', 'NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']
    #     # reduce_op = ['NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']
    #     # reduce_op = ['NP', 'NEB', 'WS', 'LAa']
    #     # now
    #     # reduce_op = ['DM', 'LE', 'LAs', 'NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']   # 53
    #     # reduce_op = ['LE', 'LAs', 'NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']  # 65
    #     # reduce_op = ['DR', 'NS', 'NP', 'NEB', 'WS', 'LAa'] # 94
    #     # reduce_op = ['LAa']   # 122
    #     reduce_op = []  # 123
    #     # for mnist
    #     # before
    #     # reduce_op = ['DR', 'DM', 'LAs', 'NAI', 'GF', 'NP', 'WS', 'LAa', 'NEB', 'NS']
    #     # reduce_op = ['NAI', 'GF', 'NP', 'WS', 'LAa', 'NEB', 'NS']
    #     # reduce_op = ['WS', 'LAa', 'NEB', 'NS']
    #     # reduce_op = ['NS']
    #     # now
    #     # reduce_op = ['LE', 'DR', 'DM', 'LAs', 'NAI', 'NP', 'GF', 'LAa', 'WS', 'NEB', 'NS']  # 47
    #     # reduce_op = ['NAI', 'NP', 'GF', 'LAa', 'WS', 'NEB', 'NS']  # 94
    #     # reduce_op = ['LAa', 'WS', 'NEB', 'NS']  # 111
    #     # reduce_op = ['LAa', 'WS', 'NEB', 'NS']  # 111
    #     # reduce_op = []  # 119
    #     # for cifar10
    #     # before
    #     # reduce_op = ['NAI', 'NS', 'LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']
    #     # reduce_op = ['NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']
    #     # reduce_op = ['LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']
    #     # reduce_op = ['DR', 'LAs', 'DF', 'LD', 'LAm']
    #     # new
    #     # reduce_op = ['NEB', 'NAI', 'NS', 'LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']  # 47
    #     # reduce_op = ['NAI', 'NS', 'LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']  # 67
    #     # reduce_op = ['LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']  # 81
    #     # reduce_op = ['DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']  # 89
    #     # reduce_op = []  # 93
    #     # for svhn
    #     # before
    #     # reduce_op = ['WS', 'AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']
    #     # reduce_op = ['LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']
    #     # reduce_op = ['DR', 'DM', 'LE', 'NP']
    #     # reduce_op = ['LE', 'NP']
    #     # new
    #     # reduce_op = ['NEB', 'WS', 'AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']  # 56
    #     # reduce_op = ['WS', 'AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']  # 71
    #     # reduce_op = ['AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']  # 85
    #     # reduce_op = ['LAs', 'DF', 'DR', 'DM', 'LE', 'NP']  # 93
    #     # reduce_op = []  # 95
    #
    #     minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)

    # 分段构建测试充分集 根据质量分数分组 在所有的类别上构建 增加实验部分
    # for num in range(10,15):
    #     kind = 'seg_class_qs'
    #     # # for lenet5
    #     # reduce_op = ['DR', 'LAs', 'WS', 'NP', 'NEB', 'GF', 'LAa']  # 101
    #     # reduce_op = ['WS', 'NP', 'NEB', 'GF', 'LAa'] # 112
    #     # reduce_op = ['NEB', 'GF', 'LAa']   # 119
    #     # reduce_op = ['GF', 'LAa']  # 123
    #     # reduce_op = [] #123
    #     # for mnist
    #     # reduce_op = ['DM', 'DF', 'WS', 'NP', 'NS', 'LAa', 'NEB', 'GF']  # 97
    #     # reduce_op = ['NP', 'NS', 'LAa', 'NEB', 'GF']  # 111
    #     # reduce_op = ['NS', 'LAa', 'NEB', 'GF']  # 116
    #     # reduce_op = ['NEB', 'GF']  # 119
    #     # reduce_op = []  # 119
    #     # for cifar10
    #     # reduce_op = ['NAI', 'LAa', 'LR', 'AFRs', 'LAs', 'DR', 'LE', 'DM', 'DF', 'NP']  # 81
    #     # reduce_op = ['AFRs', 'LAs', 'DR', 'LE', 'DM', 'DF', 'NP']  # 87
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP']  # 91
    #     # reduce_op = []  # 93
    #     # for svhn
    #     # reduce_op = ['WS', 'NAI', 'LAa', 'DM', 'AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']  # 75
    #     # reduce_op = ['NAI', 'LAa', 'DM', 'AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']  # 86
    #     reduce_op = ['NP', 'DF', 'LAs']  # 95
    #     # reduce_op = []  # 95
    #
    #     minimal_test_suit_index_reduce(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)

    # 根据质量分数对变异算子进行分类，然后根据原始测试充分集测量测试的损失 test lost  不涉及冗余的概念  在所有类上进行构建
    # for num in range(40,60):
    #     kind = 'qs_seg_random_class'
    #     reduce_op = ['DM', 'LE', 'DF', 'AFRs', 'DR', 'LAs', 'NS', 'NAI', 'NP', 'WS', 'NEB', 'GF', 'LAa']
    #     reduce_op = random.sample(reduce_op, 4)  # cifar10  9  5  1
    #     # reduce_op.append('LD')
    #     # reduce_op.append('LAm')
    #     print('aa', reduce_op)
    #     # kind = 'qs_seg_class'
    #     # for lenet5
    #     # reduce_op = ['DR', 'LAs', 'WS', 'NP', 'NEB', 'GF', 'LAa']
    #     # reduce_op = ['WS', 'NP', 'NEB', 'GF', 'LAa']
    #     # reduce_op = ['GF', 'LAa']
    #     # for mnist
    #     # reduce_op = ['NAI', 'LE', 'LAs', 'DR', 'AFRs', 'DM', 'DF', 'WS', 'NP', 'NS', 'LAa', 'NEB', 'GF']
    #     # reduce_op = ['NS', 'LAa', 'GF', 'NEB']
    #     # reduce_op = ['GF', 'NEB']
    #     # for cifar10
    #     # reduce_op = ['LAa', 'LR', 'AFRs', 'LAs', 'DR', 'LE', 'DM', 'DF', 'NP', 'LD', 'LAm']
    #     # reduce_op = ['DR', 'LE', 'DM', 'DF', 'NP', 'LD', 'LAm']
    #     # reduce_op = ['NP', 'LD', 'LAm']
    #     # for svhn
    #     # reduce_op = ['NAI', 'LAa', 'DM', 'AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']
    #     # reduce_op = ['DM', 'AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']
    #     # reduce_op = ['DR', 'NP', 'DF', 'LAs']
    #
    #     minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)


    # 依次增加一个变异算子构建测试充分集  在非冗余类别上构建   冗余分数  变异分数
    # if subject_name == 'lenet5':
    #     # op_list = ['LE', 'DM', 'DF', 'LAs', 'NS', 'DR', 'NP', 'AFRs', 'WS', 'GF', 'NAI', 'NEB', 'LAa']
    #     op_list = ['AFRs', 'DF', 'GF', 'DM', 'LE', 'LAs', 'NAI', 'DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']
    # elif subject_name == 'mnist':
    #     # op_list = ['LE', 'LAs', 'AFRs', 'DF', 'NP', 'WS', 'NAI', 'LAa', 'DR', 'DM', 'GF', 'NEB', 'NS']
    #     # op_list = ['AFRs', 'DF', 'LE', 'DR', 'DM', 'LAs', 'NAI', 'NP', 'GF', 'LAa', 'WS', 'NEB', 'NS']
    #     op_list = ['AFRs', 'DF', 'LE', 'DR', 'DM', 'GF', 'LAs', 'NAI', 'NP', 'LAa', 'WS', 'NEB', 'NS']
    # elif subject_name == 'cifar10':
    #     # op_list = ['WS', 'NEB', 'NS', 'GF', 'NAI', 'NP', 'LAa', 'LE', 'AFRs', 'DM', 'DF', 'LAs', 'LR',
    #     #            'DR', 'LD', 'LAm']
    #     op_list = ['GF', 'WS', 'NEB', 'NAI', 'NS', 'LAa', 'NP', 'DM', 'LE', 'LR', 'AFRs', 'DR', 'LAs', 'DF', 'LD', 'LAm']
    # elif subject_name == 'svhn':
    #     # op_list = ['NS', 'GF', 'NEB', 'WS', 'NAI', 'LAa', 'LAs', 'AFRs', 'DR', 'LE', 'DM', 'DF', 'NP']
    #     op_list = ['GF', 'NS', 'NEB', 'WS', 'AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP']
    # reduce_op = op_list.copy()
    # index = 5
    # for num in range(len(op_list)):
    #     print('num:', num)
    #     kind = 'single_class'
    #     reduce_op.remove(op_list[num])
    #     print('#####################:', reduce_op)
    #     for i in range(index, index + 5):
    #         minimal_test_suit_index_reduce_single_class(subject_name, mutated_model_path, predictions_path, kind, i, reduce_op)
    #     index += 5

    # 依次增加一个变异算子构建测试充分集  在所以类别上构建   质量分数  测试损失
    # if subject_name == 'lenet5':
    #     op_list = ['NS', 'DM', 'NAI', 'LE', 'DF', 'AFRs', 'DR', 'LAs', 'WS', 'NP', 'NEB', 'GF', 'LAa']
    # elif subject_name == 'mnist':
    #     op_list = ['NAI', 'LE', 'LAs', 'DR', 'AFRs', 'DM', 'DF', 'WS', 'NP', 'NS', 'LAa', 'NEB', 'GF']
    # elif subject_name == 'cifar10':
    #     op_list = ['GF', 'NEB', 'NS', 'WS', 'NAI', 'LAa', 'LR', 'AFRs', 'LAs', 'DR', 'LE', 'DM', 'DF', 'NP']
    #     # 根据质量分数约减的时候不考虑LD和LAm
    # elif subject_name == 'svhn':
    #     op_list = ['NS', 'NEB', 'GF', 'WS', 'NAI', 'LAa', 'DM', 'AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']
    # reduce_op = op_list.copy()
    # index = 0
    # for num in range(len(op_list)):
    #     print('num:', num)
    #     kind = 'single_class_qs'
    #     reduce_op.remove(op_list[num])
    #     print('#####################:', reduce_op)
    #     for i in range(index, index + 5):
    #         minimal_test_suit_index_reduce_qs_seg(subject_name, mutated_model_path, predictions_path, kind, num, reduce_op)
    #     index += 5