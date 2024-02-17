import glob
import os
import csv
from redundancy_analysis import reader_list_from_csv


def related_class(subject_name, mutated_model_path, test_killed_class_dict):
    related_class_dict = {}
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    for mutant in mutants_path:
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        related_class_dict[mutant_name] = test_killed_class_dict[mutant_name]
    return related_class_dict


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


def print_test_kill_class(subject_name, mutated_model_path, predictions_path):
    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))
    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    op_relate_class_num = {}
    sum_class_num = 0
    for mutant_name in mutant_list:
        op = mutant_name.split('_')[1]

        try:
            # op_relate_class_num[op] += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] += len(test_killed_class_dict[mutant_name])
        except Exception as e:
            # op_relate_class_num[op] = len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] = len(test_killed_class_dict[mutant_name])
        # sum_class_num += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
        sum_class_num += len(test_killed_class_dict[mutant_name])
    op_relate_class_num['all'] = sum_class_num
    return op_relate_class_num


def print_unreduntant_class(subject_name, mutated_model_path, predictions_path):
    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    op_relate_class_num = {}
    sum_class_num = 0
    for mutant_name in mutant_list:
        op = mutant_name.split('_')[1]

        try:
            # op_relate_class_num[op] += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] += len(test_killed_class_dict[mutant_name])
        except Exception as e:
            # op_relate_class_num[op] = len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] = len(test_killed_class_dict[mutant_name])
        # sum_class_num += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
        sum_class_num += len(test_killed_class_dict[mutant_name])
    op_relate_class_num['all'] = sum_class_num
    return op_relate_class_num


def calculate_mutation_score(subject_name, test_set_kind, mutated_model_path, predictions_path, time):
    print('------------------' + test_set_kind + ':' + str(time) + '------------------------')
    # mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    # mutants_num = len(mutants_path)
    # print('mutants_num:', mutants_num)

    mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'killed_mutant.csv'))
    # mutant_list = reader_list_from_csv(os.path.join(predictions_path, subject_name, 'non_redundant_mutant.csv'))

    test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)

    t_killed_class_dict = os.path.join(predictions_path, subject_name, 'result' + str(time), 'mutant', test_set_kind, 'killed_class.csv')
    t_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, t_killed_class_dict)

    reduntant_class_csv = os.path.join(predictions_path, subject_name, "reduntant_class.csv")
    reduntant_class = read_set_from_csv(mutated_model_path, subject_name, reduntant_class_csv)

    op_relate_class_num = {}
    op_relate_killed_class_num = {}
    sum_class_num = 0
    sum_killed_num = 0

    # 读取非冗余的变异体
    # non_redundancy_mutant = reader_list_from_csv(os.path.join(predictions_path, subject_name, subject_name + '_non_redundantaa.csv'))
    # for mutant_name in non_redundancy_mutant:  # 只计算在非冗余变异体上的变异分数
    # for mutant in mutants_path:    # 计算在所有非等价变异体上的变异分数
    #     mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
    for mutant_name in mutant_list:
        op = mutant_name.split('_')[1]

        try:
            # op_relate_class_num[op] += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] += len(test_killed_class_dict[mutant_name])
        except Exception as e:
            # op_relate_class_num[op] = len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
            op_relate_class_num[op] = len(test_killed_class_dict[mutant_name])
        # sum_class_num += len(test_killed_class_dict[mutant_name] - reduntant_class[mutant_name])
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
            ms = round(op_relate_killed_class_num[op] / op_relate_class_num[op], 4)
            ms_dic[op] = ms
            sum += ms
            count += 1
    ms_dic['arg'] = round(sum/count, 4)
    print('ms_dic:', ms_dic)

    return op_relate_class_num, op_relate_killed_class_num


if __name__ == '__main__':

    subject_name = 'lenet5'
    mutated_model_path = 'mutated_model_all'
    predictions_path = 'predictions_all'

    test_kill_class = print_test_kill_class(subject_name, mutated_model_path, predictions_path)
    print('op_test_kill_class_num:', test_kill_class)

    unreduntant_class = print_unreduntant_class(subject_name, mutated_model_path, predictions_path)
    print('op_unreduntant_class_num:', unreduntant_class)

    # for test_set in ['test50', 'test10', 'test10non', 'test5', 'test5non', 'diff']:
    for test_set in ['sample_reduced_class']:
    # for test_set in ['testclass']:
        all_op_relate_class_num = {}
        all_op_relate_killed_class_num = {}
        for time in range(1):
            op_relate_class_num, op_relate_killed_class_num = calculate_mutation_score(subject_name,
                                                            test_set, mutated_model_path, predictions_path, time)
            for op in op_relate_class_num.keys():
                try:
                    all_op_relate_class_num[op] += op_relate_class_num[op]
                    all_op_relate_killed_class_num[op] += op_relate_killed_class_num[op]
                except Exception as e:
                    all_op_relate_class_num[op] = op_relate_class_num[op]
                    all_op_relate_killed_class_num[op] = op_relate_killed_class_num[op]
        arg_ms_dic = {}
        sum = 0
        count = 0
        for op in all_op_relate_class_num.keys():
            if op != 'all':
                ms = round(all_op_relate_killed_class_num[op] / all_op_relate_class_num[op], 4)
                arg_ms_dic[op] = ms
                sum += ms
                count += 1
        # arg_ms_dic['arg'] = round(sum / count, 2)
        arg_ms_dic['arg'] = round(sum / count, 4)
        print('###############################' + test_set + '###############################################')
        print('arg_ms_dic:', arg_ms_dic)
        print('##########################################################################################')

    # calculate_mutation_score(subject_name, 'weak', mutated_model_path, predictions_path, 0)
    # calculate_mutation_score(subject_name, 'mki', mutated_model_path, predictions_path, 0)
    # calculate_mutation_score(subject_name, 'test50', mutated_model_path, predictions_path, time)
    # calculate_mutation_score(subject_name, 'test10', mutated_model_path, predictions_path, time)
    # calculate_mutation_score(subject_name, 'test10non', mutated_model_path, predictions_path, time)
    # calculate_mutation_score(subject_name, 'test5', mutated_model_path, predictions_path, time)
    # calculate_mutation_score(subject_name, 'test5non', mutated_model_path, predictions_path, time)
    # test_killed_class_dict = read_set_from_csv(subject_name, 'test', mutated_model_path, predictions_path)
    # weak_killed_class_dict = read_set_from_csv(subject_name, 'weak', mutated_model_path, predictions_path)
    # rr = related_class(subject_name, mutated_model_path, test_killed_class_dict, weak_killed_class_dict)
    # print(rr)
    # print('rr', rr)
    # sum = 0
    # for v in rr.keys():
    #     sum += len(rr[v])
    # print(sum)



