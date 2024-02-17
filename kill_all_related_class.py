import subprocess
from calculate_ms import read_set_from_csv
from keras.datasets import mnist, cifar10
import random
import glob
import os
import numpy as np
import h5py
from redundancy_analysis import reader_list_from_csv
import csv


def related_class(subject_name, mutated_model_path, test_killed_class_dict):
    related_class_dict = {}
    mutants_path = glob.glob(os.path.join(mutated_model_path, subject_name, '*.h5'))
    for mutant in mutants_path:
        mutant_name = (mutant.split('\\'))[-1].replace('.h5', '')
        try:
            related_class_dict[mutant_name] = test_killed_class_dict[mutant_name]
        except Exception as e:
            continue
    return related_class_dict


def is_kill_all_class(class_dict):
    sum = 0
    for mutant in class_dict:
        sum += len(class_dict[mutant])
    print('sum:', sum)
    if sum:
        return False
    else:
        return True


def kill_all_related_class(subject_name, class_dict, num, kind):
    test_x = list()
    test_y = list()
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

    original_predictions = os.path.join(predictions_path, subject_name, 'result0', 'original', 'orig_test.npy')
    ori_predict = np.load(original_predictions)
    mutants_predctions = os.path.join(predictions_path, subject_name, 'result0', 'mutant', 'test')
    index = [i for i in range(len(y_test))]
    # new_class_dict = {}
    # for mutant in non_redundancy_mutant:
    #     new_class_dict[mutant] = class_dict[mutant]
    while not is_kill_all_class(class_dict):
        ind = random.choice(index)
        index.remove(ind)
        kill = False
        for mutant in class_dict.keys():
        # for mutant in non_redundancy_mutant:
            mutant_predctions_path = os.path.join(mutants_predctions, mutant + "_test.npy")
            mutant_predict = np.load(mutant_predctions_path)

            if ori_predict[ind] == y_test[ind] and mutant_predict[ind] != y_test[ind]:
                if str(ori_predict[ind]) in class_dict[mutant]:
                    class_dict[mutant].remove(str(ori_predict[ind]))
                    kill = True
        if kill:
            test_x.append(x_test[ind])
            test_y.append(y_test[ind])
    hf = h5py.File(os.path.join('test_suit', kind, subject_name, 'data_' + str(num) + '.h5'), 'w')
    hf.create_dataset('x_test', data=test_x)
    hf.create_dataset('y_test', data=test_y)
    hf.close()
    return test_x, test_y


# 随机采样以杀死变异体所有的相关类
if __name__ == '__main__':
    # subject_name = 'lenet5'
    # mutated_model_path = 'mutated_model_WNAI'
    # predictions_path = 'predictions_WNAI'
    #
    # # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    # # non_redundancy_mutant = reader_list_from_csv(os.path.join(predictions_path, subject_name, subject_name + '_non_redundant.csv'))
    #
    # for i in range(1,5):
    #     test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    #     related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    #     print(related_class_dict)
    #     test_x, test_y = kill_all_related_class(subject_name, related_class_dict, i, 'WNAI_class')
    #     print(len(test_x))
    #     print(len(test_y))

    # mutated_model_path = 'mutated_model_random'
    # predictions_path = 'predictions_random'
    #
    # # test_killed_class_csv = os.path.join(predictions_path, subject_name, 'killed_class.csv')
    # test_killed_class_csv = os.path.join(predictions_path, subject_name, "unreduntant_class.csv")
    # # non_redundancy_mutant = reader_list_from_csv(os.path.join(predictions_path, subject_name, subject_name + '_non_redundant.csv'))
    #
    # for i in range(1):
    #     test_killed_class_dict = read_set_from_csv(mutated_model_path, subject_name, test_killed_class_csv)
    #     related_class_dict = related_class(subject_name, mutated_model_path, test_killed_class_dict)
    #     test_x, test_y = kill_all_related_class(subject_name, related_class_dict, i, 'random_class')
    #     print(len(test_x))
    #     print(len(test_y))

    for i in range(1):
        for kind in ['sample_reduced_class']:
            subprocess.run(["python", "prediction.py",
                            "-subject_name", "cifar10",
                            "-dataset", "cifar10",
                            "-mutated_model", "mutated_model_all",
                            "-predictions_path", "predictions_all",
                            "-time", str(i),
                            "-test_set_kind", kind])

    # for i in range(1):
    #     for kind in ['sample_all_class']:
    #         subprocess.run(["python", "prediction.py",
    #                         "-subject_name", "lenet5",
    #                         "-dataset", "mnist",
    #                         "-mutated_model", "mutated_model_random",
    #                         "-predictions_path", "predictions_random",
    #                         "-time", str(i),
    #                         "-test_set_kind", kind])

    # for i in range(1):
    #     for kind in ['sample_random_class']:
    #         subprocess.run(["python", "prediction.py",
    #                         "-subject_name", "lenet5",
    #                         "-dataset", "mnist",
    #                         "-mutated_model", "mutated_model_random",
    #                         "-predictions_path", "predictions_random",
    #                         "-time", str(i),
    #                         "-test_set_kind", kind])
    # for i in range(1):
    #     for kind in ['sample_all_class']:
    #         subprocess.run(["python", "prediction.py",
    #                         "-subject_name", "lenet5",
    #                         "-dataset", "mnist",
    #                         "-mutated_model", "mutated_model_all",
    #                         "-predictions_path", "predictions_all",
    #                         "-time", str(i),
    #                         "-test_set_kind", kind])
