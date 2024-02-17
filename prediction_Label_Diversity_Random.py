
import h5py
import numpy as np
import os
import argparse
from keras.datasets import mnist, cifar10
from utils import load_mnist, load_cifar10, load_svhn
from tensorflow.keras.models import load_model
import glob
import keras.backend as K
import gc


def prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='cifar10',
                        help='subject name')
    # 原始模型训练参数
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_all',
                        help='mutated model saved path')
    parser.add_argument('-predictions_path',
                        type=str,
                        default='predictions_part_rs_seg_random',
                        help='predictions_path')
    parser.add_argument('-dataset',
                        type=str,
                        default='cifar10',
                        help='mnist or cifar10 or svhn')

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    predictions_path = args.predictions_path
    dataset = args.dataset

    print("aaaaaaaaaaaa：", dataset)

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = y_test.flatten()
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
        y_test = y_test.flatten()

    # 处理要加载的数据
    error_result_list = []
    sum = 0
    for ii in range(220, 240):
        hf = h5py.File(os.path.join('test_suit_newexp', 'random_seg_class_rs', subject_name, 'index_' + str(ii) + '.h5'), 'r')
        index = np.asarray(hf.get('index'))
        print(index)

        xx_test = list()
        for i in index:
            xx_test.append(x_test[i])

        yy_test = y_test[index]
        xx_test = np.asarray(xx_test)

        # print(xx_test)
        print(len(xx_test))
        # print(yy_test)
        print(len(yy_test))

        # 在原始模型上进行预测
        original_model_path = os.path.join(original_model, subject_name + '_original.h5')
        original_predictions = os.path.join(predictions_path, subject_name, 'result' + str(ii), 'original',
                                            'orig_test.npy')

        if not os.path.exists(os.path.join(predictions_path, subject_name, 'result' + str(ii), 'original')):
            try:
                os.makedirs(os.path.join(predictions_path, subject_name, 'result' + str(ii), 'original'))
            except OSError as e:
                print('Unable to create folder for analysis results:' + str(e))

        if not os.path.exists(original_predictions):
            ori_model = load_model(original_model_path)
            ori_predict = ori_model.predict(xx_test).argmax(axis=-1)

            np.save(original_predictions, ori_predict)
        else:
            ori_predict = np.load(original_predictions)

        print("yy_test:", yy_test)
        print("ori_predict", ori_predict)

        mutants_predctions = os.path.join(predictions_path, subject_name, 'result' + str(ii), 'mutant', 'test')

        if not os.path.exists(mutants_predctions):
            try:
                os.makedirs(mutants_predctions)
            except OSError as e:
                print('Unable to create folder for analysis results:' + str(e))
        mutants_path = glob.glob(os.path.join(mutated_model, subject_name, '*.h5'))
        mutants_num = len(mutants_path)
        print('mutants_num:', mutants_num)

        error_label_nums = np.zeros((10, 10))

        for mutant in mutants_path:
            mutant_name = (mutant.split("\\"))[-1].replace(".h5", "")
            mutant_predctions_path = os.path.join(mutants_predctions, mutant_name + "_test.npy")
            if not os.path.exists(mutant_predctions_path):
                print("当前执行的变异体：", mutant)
                model = load_model(mutant)
                result = model.predict(xx_test).argmax(axis=-1)
                np.save(mutant_predctions_path, result)
                K.clear_session()
                del model
                gc.collect()
            else:
                result = np.load(mutant_predctions_path)
            print("mutant_predict:", result)
            error_index = np.where(ori_predict != result)[0]
            for ei in error_index:
                error_label_nums[ori_predict[ei]][result[ei]] = 1
            print("########:", np.sum(error_label_nums))
        error_result_list.append(np.sum(error_label_nums))
        sum += np.sum(error_label_nums)
        print("error_result_list:", error_result_list)
    print("error_result_list:", error_result_list)
    print("sum:", sum)
    print("avg:", round(sum/20, 2))

# 结果
# rs vs random
# lenet5
# 56 [33.0, 49.0, 39.0, 37.0, 41.0, 31.0, 27.0, 17.0, 46.0, 49.0, 49.0, 29.0, 30.0, 41.0, 40.0, 53.0, 42.0, 30.0, 38.0, 40.0] 38.05
# 63 [55.0, 53.0, 36.0, 43.0, 42.0, 49.0, 42.0, 46.0, 33.0, 55.0, 51.0, 56.0, 43.0, 53.0, 50.0, 60.0, 57.0, 51.0, 44.0, 54.0] 48.65
# 75 [67.0, 58.0, 50.0, 62.0, 63.0, 56.0, 70.0, 60.0, 65.0, 62.0, 56.0, 62.0, 67.0, 74.0, 56.0, 61.0, 64.0, 57.0, 66.0, 55.0] 61.55
# 77 [76.0, 76.0, 73.0, 74.0, 75.0, 76.0, 76.0, 77.0, 76.0, 71.0, 75.0, 76.0, 76.0, 76.0, 77.0, 76.0, 73.0, 76.0, 76.0, 77.0] 75.40
# mnist
# 58 [25.0, 9.0, 39.0, 58.0, 12.0, 22.0, 32.0, 16.0, 26.0, 25.0, 21.0, 34.0, 5.0, 13.0, 8.0, 8.0, 58.0, 14.0, 23.0, 32.0] 24.00
# 71 [63.0, 51.0, 66.0, 44.0, 62.0, 49.0, 54.0, 55.0, 46.0, 59.0, 45.0, 65.0, 59.0, 60.0, 62.0, 63.0, 63.0, 61.0, 52.0, 43.0] 56.10
# 74 [73.0, 50.0, 70.0, 65.0, 67.0, 70.0, 72.0, 68.0, 68.0, 69.0, 56.0, 60.0, 69.0, 55.0, 67.0, 67.0, 63.0, 51.0, 69.0, 65.0] 64.70
# svhn
# 83 [71.0, 77.0, 77.0, 80.0, 76.0, 48.0, 47.0, 72.0, 49.0, 80.0, 76.0, 42.0, 46.0, 56.0, 51.0, 56.0, 48.0, 49.0, 51.0, 51.0] 60.15
# 86
# cifar10
# 79 [63.0, 35.0, 49.0, 73.0, 73.0, 68.0, 43.0, 55.0, 60.0, 49.0, 73.0, 74.0, 48.0, 69.0, 64.0, 76.0, 73.0, 64.0, 46.0, 44.0] 59.95
# 81 [85.0, 64.0, 76.0, 43.0, 56.0, 82.0, 76.0, 62.0, 46.0, 66.0, 81.0, 75.0, 46.0, 67.0, 41.0, 48.0, 75.0, 37.0, 39.0, 78.0] 62.15
# 87 [86.0, 65.0, 79.0, 87.0, 84.0, 69.0, 80.0, 78.0, 67.0, 74.0, 77.0, 83.0, 89.0, 76.0, 80.0, 65.0, 80.0, 78.0, 82.0, 46.0] 76.25
# 88 [80.0, 85.0, 81.0, 83.0, 87.0, 79.0, 82.0, 87.0, 79.0, 73.0, 83.0, 84.0, 83.0, 86.0, 83.0, 89.0, 84.0, 87.0, 72.0, 85.0] 82.60

# qs vs random
# lenet5
# 75 [75.0, 71.0, 70.0, 70.0, 75.0, 74.0, 71.0, 69.0, 71.0, 68.0, 73.0, 73.0, 69.0, 73.0, 72.0, 71.0, 71.0, 69.0, 68.0, 72.0] 71.25
# 75


if __name__ == '__main__':
    prediction()
