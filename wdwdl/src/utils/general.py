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


def train_test_ids_from_data_set(data_set, label, test_size):
    """
    Splits a data set into a train set and a test set.
    :param data_set:
    :param label:
    :param test_size:
    :return: ids for train and test set.
    """
    return sklearn.model_selection.train_test_split(data_set, label, test_size=test_size, shuffle=True)


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
    if Path(args.result_dir + file_type + "_" + args.data_set).is_file():
        pass
        # open('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4]), "w").close()
    else:
        file = open('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4]), "w+")
        if file_type == "metrics":
            file.write("Dataset; accuracy; f1_score; precision; recall; auc_roc")
        file.close()


def add_to_file(args, file_type, input_):
    file = open('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4]), "a+")
    if file_type == "metrics":

        file.write("\n" + args.data_set[:-4] + ";" +
                   str(input_["accuracy"][-1]) + ";" +
                   str(input_["f1_score"][-1]) + ";" +
                   str(input_["precision"][-1]) + ";" +
                   str(input_["recall"][-1]) + ";" +
                   str(input_["auc_roc"][-1]))

    elif file_type == "hyper_params":

        if os.stat('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4])).st_size != 0:
            file.write("\n")

        file.write("Dataset: " + str(args.data_set[:-4]) + "\n")
        file.write("Best f1-score value: " + str(round(input_.value, 4)) + "\n")
        file.write("Best params:\n")

        for key, value in input_.params.items():
            file.write("%s: %s\n" % (str(key), str(value)))

    elif file_type == "predictions":

        if os.stat('%s%s_%s.csv' % (args.result_dir, file_type, args.data_set[:-4])).st_size != 0:
            file.write("\n")

        file.write("Dataset: " + str(args.data_set[:-4]) + "\n")
        file.write("Predictions: " + str(input_))

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


