from model_level_operators import *
from tensorflow.keras.models import load_model
import argparse
from utils_model_op import summary_model, color_preprocessing, model_predict
from termcolor import colored
from keras.datasets import mnist, cifar10
import gc
import keras.backend as K
from progressbar import *
from utils import load_svhn


def model_level_mutants_generation(ori_model, operator, ratio, standard_deviation=0.5):
    """
    :param ori_model:
    :param operator:
    :param ratio:
    :param standard_deviation:
    :return:
    """
    if operator < 5:
        model_level_operators(ori_model, operator, ratio, standard_deviation)
    else:
        new_model = model_level_operators(ori_model, operator, ratio, standard_deviation)
        return new_model
    return ori_model


def generator():
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument('-subject_name',
                        type=str,
                        default='lenet5',
                        help='subject name')
    parser.add_argument('-original_model',
                        type=str,
                        default='original_model',
                        help='original model saved path')
    parser.add_argument('-mutated_model',
                        type=str,
                        default='mutated_model_all',
                        help='mutated model saved path')
    parser.add_argument('-model_nums',
                        type=int,
                        default=20)
    parser.add_argument('-dataset',
                        type=str,
                        default="mnist",
                        help="mnist or cifar10 or svhn")
    parser.add_argument('-operator',
                        type=int,
                        default=7,
                        help="mutation operator 0-GF 1-WS 2-NEB 3-NAI 4-NS 5-LD 层失活 6-LAa  7-LAm 添加层(目前的代码实现的是复制层)")
    parser.add_argument('-ratio',
                        type=float,
                        default=0.01,
                        help="ratio of important neurons to be mutated lenet5/mnist-0.01 svhn/cifar10-0.003")
    parser.add_argument('-standard_deviation',
                        type=float,
                        default=0.5,
                        help="standard_deviation for gaussian fuzzing")
    parser.add_argument('-threshold',
                        type=float,
                        default=0.9,
                        help="ori acc * threshold must > mutants acc")

    args = parser.parse_args()
    subject_name = args.subject_name
    mutated_model = args.mutated_model
    original_model = args.original_model
    model_nums = args.model_nums
    dataset = args.dataset
    operator = args.operator
    ratio = args.ratio
    standard_deviation = args.standard_deviation
    threshold = args.threshold

    # load data
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = color_preprocessing(x_train, x_test, 0, 255)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
    elif dataset == 'cifar10':
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # x_train, x_test = color_preprocessing(x_train, x_test, [125.307, 122.95, 113.865], [62.9932, 62.0887, 66.7048])
        # x_test = x_test.reshape(len(x_test), 32, 32, 3)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()

    mutated_model_path = os.path.join(mutated_model, subject_name, '')
    if not os.path.exists(mutated_model_path):
        try:
            os.makedirs(mutated_model_path)
        except OSError as e:
            print('Unable to create folder for analysis results:' + str(e))

    original_model_path = os.path.join(original_model, subject_name + '_original')
    ori_model_path = original_model_path + '.h5'
    ori_model = load_model(ori_model_path)
    ori_model.summary()
    ori_acc = model_predict(ori_model, x_test, y_test)
    threshold = ori_acc * threshold

    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(ori_model)
    print(colored("operator: %s" % model_level_operator_name(operator), 'blue'))
    print(colored("ori acc: %f" % ori_acc, 'blue'))
    print(colored("threshold acc: %f" % threshold, 'blue'))
    if operator == 0 or operator == 1:
        print("total weights: ", weight_count)
        print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)
    elif 2 <= operator <= 4:
        print("total neuron: ", neuron_count)
        print("process neuron num: ", int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1)

    # mutants generation
    p_bar = ProgressBar().start()
    i = 1
    start_time = time.clock()
    while i <= model_nums:
        model_save_path = subject_name + '_' + model_level_operator_name(operator) + '_' + str(ratio) + '_mutated_' + str(i) + '.h5'
        if not os.path.exists(mutated_model_path + model_save_path):
            original_model = load_model(ori_model_path)
            new_model = model_level_mutants_generation(original_model, operator, ratio, standard_deviation)
            new_acc = model_predict(new_model, x_test, y_test)
            print('new_acc:', new_acc)
            if new_acc < threshold:
                K.clear_session()
                del original_model
                del new_model
                gc.collect()
                continue
            new_model.save(mutated_model_path + model_save_path)
            p_bar.update(int((i / model_nums) * 100))
            i += 1
            K.clear_session()
            del original_model
            del new_model
            gc.collect()

    p_bar.finish()
    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)


if __name__ == '__main__':
    generator()

