import argparse
import os
from neurons_analysis import neural_analysis, get_trainable_layers, random_choice_neuron_non
from fnn_operator_non import fnn_operator_name
from neuron_analysis_utils import load_model, summary_model, summary_model_all, model_predict, load_mnist, load_cifar10, load_svhn
from fnn_operator_non import fnn_operator
import math
import numpy as np
import gc
import keras.backend as K


def runner():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='lenet5',
                        help='subject name')
    # 原始模型训练参数
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_random',
                        help='mutated model saved path')
    parser.add_argument('-model_nums',
                        type=int,
                        default=20)
    # 重要神经元分析参数
    parser.add_argument('-dataset',
                        type=str,
                        default="mnist",
                        help="mnist or cifar10 or svhn")
    parser.add_argument('-relevance_score_path',
                        type=str,
                        default='relevance_score',
                        help="relevance score save path")
    # 模型变异参数
    parser.add_argument('-operator',
                        type=int,
                        default=2,
                        help="mutation operator 0-GF 1-NEB 2-NAI 3-WS 4-NS 7-NIS")
    parser.add_argument('-ratio',
                        type=float,
                        default=0.01,
                        help="ratio of important neurons to be mutated")
    parser.add_argument('-standard_deviation',
                        type=float,
                        default=1.0,
                        help="standard_deviation for gaussian fuzzing")
    parser.add_argument('-threshold',
                        type=float,
                        default=0.9,
                        help="")

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    model_nums = args.model_nums
    dataset = args.dataset
    relevance_score_path = args.relevance_score_path
    # subject_layer = args.subject_layer
    operator = args.operator
    ratio = args.ratio
    standard_deviation = args.standard_deviation
    threshold = args.threshold

    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
    else:
        x_train, y_train, x_test, y_test = load_svhn(one_hot=True)

    mutated_model_path = os.path.join(mutated_model, subject_name, '')
    if not os.path.exists(mutated_model_path):
        try:
            os.makedirs(mutated_model_path)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    original_model_path = os.path.join(original_model, subject_name + '_original')
    ori_model_path = original_model_path + '.h5'
    ori_model = load_model(ori_model_path, subject_name, dataset)
    ori_acc = ori_model.evaluate(x_test, y_test, verbose=0)[1]
    print('aaaaaaaaaaaa:', ori_acc)
    threshold = ori_acc * threshold
    trainable_layers = get_trainable_layers(ori_model)
    untrainable_layers = list(set(range(len(ori_model.layers))) - set(trainable_layers))
    print('trainable_layer:', trainable_layers)
    print('untrainable_layer:', untrainable_layers)
    weight_count_all, neuron_count_all, weights_dict_all, neuron_dict_all = summary_model_all(ori_model)
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(ori_model, trainable_layers)
    print(neuron_dict_all)
    print(neuron_dict)

    # 对模型中不重要的神经元进行变异
    i = 0
    while i < model_nums:
        model_save_path = subject_name + '_' + fnn_operator_name(operator) + '_' + str(ratio) + '_mutated_random_' + str(i) + '.h5'
        # 如果变异的模型模型不存在
        if not os.path.exists(mutated_model_path + model_save_path):
            # target_num = math.ceil(neuron_count * ratio)
            target_num = math.ceil(neuron_count_all * ratio)
            process_neuron_num = target_num
            print('process_neuron_num:', process_neuron_num)

            # 每层要处理的重要神经元的数量
            process_num_dict = random_select(neuron_count, process_neuron_num, neuron_dict)
            print('process_num_dict:', process_num_dict)

            # 从每层中随机选择要变异的神经元
            l_non_important_neurons = random_choice_neuron_non(ori_model, neuron_dict, process_num_dict)
            original_model = load_model(ori_model_path, subject_name, dataset)
            new_model = fnn_operator(original_model, ori_model_path, operator, l_non_important_neurons, relevance_score_path, standard_deviation)
            new_acc = new_model.evaluate(x_test, y_test, verbose=0)[1]
            print('new_acc:', new_acc)
            if new_acc < threshold:
                K.clear_session()
                del original_model
                del new_model
                gc.collect()
                continue
            new_model.save_weights(mutated_model_path + model_save_path)
            i += 1


def random_select(neurons_num, process_neuron_num, neuron_dict):
    # 每层要处理的神经元的数量
    process_num_dict = {}
    process_num_total = 0
    ll = 0
    indices = np.random.choice(neurons_num, process_neuron_num, replace=False)
    for i, layer_name in enumerate(neuron_dict.keys()):
        if i == 0:
            num = len(np.where(indices < neuron_dict[layer_name])[0])
            process_num_dict[layer_name] = num
            process_num_total += num
            ll = neuron_dict[layer_name]
        else:
            num = len(np.where(indices < ll + neuron_dict[layer_name])[0])
            num -= process_num_total
            process_num_total += num
            process_num_dict[layer_name] = num
            ll += neuron_dict[layer_name]
    return process_num_dict


if __name__ == '__main__':
    # 运行
    runner()

