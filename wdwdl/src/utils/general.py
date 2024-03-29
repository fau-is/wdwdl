import argparse
import sys
import numpy as np
import sklearn
import os
import tensorflow as tf
import random
from pathlib import Path


class_names = ["No workaround",
               "Injured responsibility",
               "Manipulated data",
               "Repeated activity",
               "Substituted activity",
               "Interchanged activity",
               "Bypassed activity",
               "Added activity"
               ]


def set_seed(args):
    """
    Sets seed for reproducible results.
    :param args: args.
    :return: none.
    """
    np.random.seed(args.seed_val)
    tf.random.set_seed(args.seed_val)
    random.seed(args.seed_val)


def train_test_sets_from_data_set(args, data_set, label, test_size):
    """
    Returns for a data set a train set and a test set.
    :param args:
    :param data_set:
    :param label:
    :param test_size:
    :return: ids for train and test set.
    """
    return sklearn.model_selection.train_test_split(data_set, label, test_size=test_size, shuffle=args.shuffle)


def train_test_ids_from_data_set(args, input_data):
    """
    Returns for a data set a set of train ids and test ids.
    :param args:
    :param input_data:
    :return:
    """

    if args.shuffle:

        shuffle_split = sklearn.model_selection.ShuffleSplit(n_splits=1, test_size=1-args.split_rate_test_hpo)

        hpo_train_indices = []
        hpo_test_indices = []

        for train_indices, test_indices in shuffle_split.split(input_data):
            hpo_train_indices.extend(train_indices.tolist())
            hpo_test_indices.extend(test_indices.tolist())

        return hpo_train_indices, hpo_test_indices

    else:
        indices_ = [index for index in range(0, len(input_data))]
        return indices_[:int(len(indices_) * args.split_rate_test_hpo)], indices_[int(len(indices_) * args.split_rate_test_hpo):]


def arg_max(list_):
    """
    Picks out the index of each row of a matrix with the highest value.
    :param list_:
    :return: list of class ids.
    """
    return np.argmax(list_, axis=1)


def convert_label_to_categorical(label):
    return tf.keras.utils.to_categorical(label)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def ams_grad():
    """
    Variant of Adam algorithm
    Ensures non-increasing step size

    See paper the following paper for more details:
    "On the Convergence of Adam and beyond" from Reddi et al. (2019).

    :return:
    """
    return tf.keras.optimizers.Adam(amsgrad=True)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_files(args):
    clear_file(args, "metrics")
    clear_file(args, "hyper_params")
    clear_file(args, "predictions")


def clear_file(args, file_type):
    file_path_ae = '%s%s_ae_%s' % (args.result_dir, file_type, args.data_set[:-4])
    file_path_dnn = '%s%s_%s' % (args.result_dir, file_type, args.data_set[:-4])

    # autoencoder
    if Path(file_path_ae).is_file():
        pass
    else:
        if file_type == 'hyper_params':
            open(file_path_ae + '.csv', "w+").close()
    # dnn
    if Path(file_path_dnn).is_file():
        pass
        # open('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4]), "w").close()
    else:
        file = open(file_path_dnn + '.csv', "w+")
        if file_type == "metrics":
            file.write(
                "Dataset; accuracy; f1_score; precision; recall; auc_roc; runs; shuffle; remove_noise; remove_noise_factor")
        file.close()


def add_to_file(args, file_type, input_, is_ae=False):
    if is_ae:
        file_name = '%s%s_ae_%s.csv' % (args.result_dir, file_type, args.data_set[:-4])
    else:
        file_name = '%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4])

    file = open(file_name, "a+")

    if file_type == "metrics":

        file.write("\n" + args.data_set[:-4] + ";" +
                   str(input_["accuracy"][-1]) + ";" +
                   str(input_["f1_score"][-1]) + ";" +
                   str(input_["precision"][-1]) + ";" +
                   str(input_["recall"][-1]) + ";" +
                   str(input_["auc_roc"][-1]) + ";" +
                   str(args.hpo_eval_runs) + ";" +
                   str(args.shuffle) + ";" +
                   str(args.remove_noise) + ";" +
                   str(args.remove_noise_threshold)
                   )
    elif file_type == "hyper_params":

        if os.stat(file_name).st_size != 0:
            file.write("\n")

        file.write("Dataset: " + str(args.data_set[:-4]) + "\n")
        file.write("Best f1-score value: " + str(round(input_.value, 4)) + "\n")
        file.write("Best params:\n")

        for key, value in input_.params.items():
                file.write("%s: %s\n" % (str(key), str(value)))

    elif file_type == "predictions":

        if os.stat(file_name).st_size != 0:
            file.write("\n")

        file.write("Dataset: " + str(args.data_set[:-4]) + "\n")
        file.write("Predictions: " + str(input_) + "\n")
        file.write("Remove noise: " + str(args.remove_noise) + "\n")
        file.write("Remove noise threshold: " + str(args.remove_noise_threshold))

    file.close()


def delete_encoders():
    folder = './encoder'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


