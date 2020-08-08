import argparse
import sys
import numpy as np
import sklearn
import os
import tensorflow as tf
import random


output = {
    "accuracy_values": [],
    "accuracy_value": 0.0,
    "precision_values": [],
    "precision_value": 0.0,
    "recall_values": [],
    "recall_value": 0.0,
    "f1_values": [],
    "f1_value": 0.0,
    "auc_roc_values": [],
    "auc_roc_value": 0.0,
    "auc_prc_values": [],
    "auc_prc_value": 0.0,
    "training_time_seconds": []
}

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open('./results/output_%s.csv' % (args.data_set[:-4]), "w").close()


def delete_encoders():
    folder = './encoder'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


