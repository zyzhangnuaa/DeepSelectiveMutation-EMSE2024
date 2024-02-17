
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
                        default='predictions_part_rs_seg',
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
    ii = 15
    hf = h5py.File(os.path.join('test_suit_newexp', 'seg_class_rs', subject_name, 'index_' + str(ii) + '.h5'), 'r')
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
    original_predictions = os.path.join(predictions_path, subject_name, 'result' + str(ii), 'original', 'orig_test.npy')

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


if __name__ == '__main__':
    prediction()
