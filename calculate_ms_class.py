import glob
import os
import csv
from redundancy_analysis import reader_list_from_csv


def read_set_from_csv(mutated_model_path, subject_name, killed_class_csv):
    test_killed_class_dict = {}
    with open(killed_class_csv) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            test_killed_class_dict[row[0]] = row[1]
    # test_killed_class_num = int(test_killed_class_dict['killed_num'])
    # print('test_killed_class_num:', test_killed_class_num)
    # 将读取的结果转换成集合
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    for mutant in mutants_path:
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        try:
            if test_killed_class_dict[mutant_name] != 'set()':
                test_killed_class_dict[mutant_name] = set(
                    test_killed_class_dict[mutant_name][1:-1].replace(' ', '').split(','))
            else:
                test_killed_class_dict[mutant_name] = set()
        except Exception as e:
            continue
    return test_killed_class_dict


def calculate_mutation_score(subject_name, mutated_model_path, predictions_path, reduced_op):

    # mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))

    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    # print('llll:', test_killed_class_dict)

    kill_num_of_class = {t: 0 for t in range(10)}
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    mutant_num = 0
    for mutant in mutants_path:
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        op = mutant_name.split('_')[1]
        if op in reduced_op:
            continue
        mutant_num += 1
        set = test_killed_class_dict[mutant_name]
        for s in set:
            kill_num_of_class[int(s)] += 1
    print('mutant_num:', mutant_num)
    ms_class = {}
    count = 0
    for key in kill_num_of_class.keys():
        ms_class[key] = round(kill_num_of_class[key]*10/(mutant_num),2)
        count += kill_num_of_class[key]

    kill_num_of_class['count'] = count

    return kill_num_of_class, ms_class


if __name__ == '__main__':

    subject_name = 'cifar10'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'
    kill_num_of_class, ms_class = calculate_mutation_score(subject_name, mutated_model_path, predictions_path,
                                                           reduced_op=[])
    print('all:', kill_num_of_class)
    print('all:', ms_class)

    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'
    # reduced_op = ['DR', 'NS', 'NP', 'NEB', 'WS', 'LAa']  # lenet5
    # reduced_op = ['NAI', 'NP', 'GF', 'LAa', 'WS', 'NEB', 'NS']  # mnist
    # reduced_op = ['AFRs', 'NAI', 'LAa', 'LAs', 'DF', 'DR', 'DM', 'LE', 'NP'] # svhn
    # reduced_op = ['DR', 'DM', 'LE', 'NP', 'DF', 'LAs'] # svhn
    # reduced_op = []
    reduced_op = ['AFRs', 'LE', 'DR', 'NP', 'DF', 'LAs']  # svhn
    # reduced_op = ['AFRs', 'LAs', 'DR', 'LE', 'DM', 'DF', 'NP', 'LD', 'LAm']
    kill_num_of_class, ms_class = calculate_mutation_score(subject_name, mutated_model_path, predictions_path,
                                                           reduced_op)
    print('reduced:', kill_num_of_class)
    print('reduced:', ms_class)




