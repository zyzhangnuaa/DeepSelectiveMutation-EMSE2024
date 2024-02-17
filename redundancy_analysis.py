import os
import numpy as np
import glob
import networkx as nx
import csv
import pandas as pd


def write_list_to_csv(csv_file, value_list):
    with open(csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for node in value_list:
            writer.writerow([node])


def reader_list_from_csv(csv_file):
    list = []
    with open(csv_file, 'r') as f1:
        reader = csv.reader(f1)
        for row in reader:
            list.append(row[0])
    return list


# 查找非冗余的可杀死类别
def analysis_redundancy(predictions_path, subject_name):
    mutant_class_input_info_path = os.path.join(predictions_path, subject_name, 'class_input_info.csv')
    all_mutant_class_input_killing_info = pd.read_csv(mutant_class_input_info_path)
    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))
    print(len(mutant_list))
    all_dict_redundant = {}
    all_dict_non_redundant = {}
    non_redundant_mutant = []
    redundant_mutant = []
    for i in range(10):
        g = nx.DiGraph()
        for mutant_prefix1 in mutant_list:
            if len(set(list(eval(all_mutant_class_input_killing_info[mutant_prefix1][i])))) == 0:
                continue
            if mutant_prefix1 not in g:
                g.add_node(mutant_prefix1)
                for mutant_prefix2 in mutant_list:
                    if mutant_prefix2 != mutant_prefix1 and len(
                            set(list(eval(all_mutant_class_input_killing_info[mutant_prefix2][i])))) != 0:
                        # print('ccc:', mutant_prefix1, mutant_prefix2)
                        if set(list(eval(all_mutant_class_input_killing_info[mutant_prefix1][i]))).issubset(
                                set(list(eval(all_mutant_class_input_killing_info[mutant_prefix2][i])))):
                            # print(set(list(eval(all_mutant_class_input_killing_info[mutant_prefix1][i]))),
                            #       set(list(eval(all_mutant_class_input_killing_info[mutant_prefix2][i]))))
                            # print('add:', mutant_prefix1, 'to', mutant_prefix2)
                            g.add_edge(mutant_prefix1, mutant_prefix2)
        try:
            nodes = np.sort(g.nodes())
        except Exception as e:
            print('aaaaaaaaaaaaaa 节点为空')
            all_dict_redundant[i] = []
            all_dict_non_redundant[i] = []
            continue
        redundant_nodes = []
        non_redundant_nodes = []

        for node in nodes:
            if g.in_degree(node) == 0:
                non_redundant_nodes.append(str(node))
                # non_redundant_mutant.add(str(node))
            else:
                redundant_nodes.append(str(node))
                # redundant_mutant.add(str(node))

        all_dict_redundant[i] = redundant_nodes
        all_dict_non_redundant[i] = non_redundant_nodes
        # print('aa:', len(all_dict_redundant[i]))
        # print('bb:', len(all_dict_non_redundant[i]))

    all_mutant_class_red_info = {}  # 记录所有变异体的冗余情况
    red_num = 0
    for mutant in mutant_list:
        all_mutant_class_red_info[mutant] = set()
    for i in range(10):
        info = all_dict_redundant[i]
        red_num += len(info)
        for mutant in info:
            all_mutant_class_red_info[mutant].add(i)

    all_mutant_class_non_red_info = {}  # 记录所有变异体的冗余情况
    nonred_num = 0
    for mutant in mutant_list:
        all_mutant_class_non_red_info[mutant] = set()
    for i in range(10):
        info = all_dict_non_redundant[i]
        nonred_num += len(info)
        for mutant in info:
            all_mutant_class_non_red_info[mutant].add(i)

    for mutant in mutant_list:
        if len(all_mutant_class_non_red_info[mutant]) != 0:
            non_redundant_mutant.append(mutant)
        else:
            redundant_mutant.append(mutant)

    return all_mutant_class_red_info, all_mutant_class_non_red_info, red_num, nonred_num, \
           non_redundant_mutant, redundant_mutant


if __name__ == '__main__':
    subject_name = 'lenet5'
    predictions_path = 'predictions_all'
    all_mutant_class_red_info, all_mutant_class_non_red_info, red_num, nonred_num, non_redundant_mutant, redundant_mutant \
        = analysis_redundancy(predictions_path, subject_name)
    red_csv_file = os.path.join(predictions_path, subject_name, "reduntant_class.csv")
    unred_csv_file = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    non_redundant_mutant_csv_file = os.path.join(predictions_path, subject_name, "non_redundant_mutant.csv")
    redundant_mutant_csv_file = os.path.join(predictions_path, subject_name, "redundant_mutant.csv")

    print('non_reduntant_class_num', nonred_num)

    with open(red_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['mutant', 'reduntant_class'])
        for key, value in all_mutant_class_red_info.items():
            writer.writerow([key, value])
        writer.writerow(['red_num', red_num])

    with open(unred_csv_file, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow(['mutant', 'non_reduntant_class'])
        for key, value in all_mutant_class_non_red_info.items():
            writer.writerow([key, value])
        writer.writerow(['non_red_num', nonred_num])

    write_list_to_csv(non_redundant_mutant_csv_file, non_redundant_mutant)
    write_list_to_csv(redundant_mutant_csv_file, redundant_mutant)