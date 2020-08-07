import argparse
import pickle
import sys
import numpy
import sklearn
import matplotlib.pyplot as pyplot
import random
import seaborn as sns
import pandas
from functools import reduce
import os
import tensorflow.keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt

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


def train_test_ids_from_data_set(data_set, label, test_size):
    """
    Splits a data set into a train set and a test set.
    :param data_set:
    :param label:
    :param test_size:
    :return: ids for train and test set.
    """
    return sklearn.model_selection.train_test_split(data_set, label, test_size=test_size, random_state=0)


def arg_max(list_):
    return numpy.argmax(list_, axis=1)


def plot_learning_curve(history, learning_epochs):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(range(learning_epochs), loss, 'bo', label='Training loss')
    plt.plot(range(learning_epochs), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



def convert_label_to_categorical(label):
    return tf.keras.utils.to_categorical(label)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open('./results/output_%s.csv' % (args.data_set[:-4]), "w").close()


def calculate_and_print_output(label_ground_truth, label_prediction):
    """
    This function calculate and prints the measures.
    """

    numpy.set_printoptions(precision=3)

    label_ground_truth = numpy.array(label_ground_truth)
    label_prediction = numpy.array(label_prediction)

    llprint("Accuracy: %f\n" % sklearn.metrics.accuracy_score(label_ground_truth, label_prediction))
    llprint("Precision: %f\n" % sklearn.metrics.precision_score(label_ground_truth, label_prediction, average='macro'))
    llprint("Recall: %f\n" % sklearn.metrics.recall_score(label_ground_truth, label_prediction, average='macro'))
    llprint("F1-score: %f\n" % sklearn.metrics.f1_score(label_ground_truth, label_prediction, average='macro'))
    llprint("Auc-roc: %f\n" % multi_class_roc_auc_score(label_ground_truth, label_prediction))
    # we use the average precision at different threshold values as auc of the pr-curve
    # and not the auc-pr-curve with the trapezoidal rule / linear interpolation, because it could be too optimistic
    llprint("Auc-prc: %f\n" % multi_class_prc_auc_score(label_ground_truth, label_prediction))


def plot_confusion_matrix2(label_ground_truth, label_prediction, args):
    label_ground_truth = numpy.array(label_ground_truth)
    label_prediction = numpy.array(label_prediction)

    classes = sklearn.utils.multiclass.unique_labels(label_ground_truth, label_prediction)
    classes_ = []
    for element in classes:
        classes_.append(class_names[element])
    classes = classes_

    cms = []
    cm = sklearn.metrics.confusion_matrix(label_ground_truth, label_prediction)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        if n > 1000000:
            return str(numpy.round(n / 1000000, 1)) + 'M'
        elif n > 1000:
            return str(numpy.round(n / 1000, 1)) + 'K'
        else:
            return str(n)

    cm = reduce(lambda x, y: x.add(y, fill_value=0), cms)
    annot = cm.applymap(prettify)
    cm = (cm.T / cm.sum(axis=1)).T
    fig, g = pyplot.subplots(figsize=(7, 4.5))
    g = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, rasterized=True, linewidths=0.1)
    _ = g.set(ylabel='Actual', xlabel='Prediction')

    for _, spine in g.spines.items():
        spine.set_visible(True)


    pyplot.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(str(args.result_dir + 'cm.pdf'))

    print(label_prediction)
    print(label_ground_truth)


def plot_confusion_matrix(label_ground_truth, label_prediction,
                          normalize=False,
                          title=None,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    numpy.set_printoptions(precision=3)

    label_ground_truth = numpy.array(label_ground_truth)
    label_prediction = numpy.array(label_prediction)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    cm = sklearn.metrics.confusion_matrix(label_ground_truth, label_prediction)

    # todo: map to class_names -> classes[]
    classes = sklearn.utils.multiclass.unique_labels(label_ground_truth, label_prediction)
    classes_ = []
    for element in classes:
        classes_.append(class_names[element])
    classes = classes_

    cm = (cm.T / cm.sum(axis=1)).T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    pyplot.show()
    return ax


def multi_class_prc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.average_precision_score(ground_truth_label, predicted_label, average=average)


def multi_class_roc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.roc_auc_score(ground_truth_label, predicted_label, average=average)


def get_unique_events(process_instances_):
    flat_list = [item for sublist in process_instances_ for item in sublist]
    unique_events = numpy.unique(numpy.array(flat_list))

    return unique_events


def get_unique_context(process_instances_context_):
    num_context_attr = len(process_instances_context_[0][0])
    unique_context = []

    for index in range(0, num_context_attr):

        flat_list = [item[index] for sublist in process_instances_context_ for item in sublist]
        unique_context_attr = numpy.unique(numpy.array(flat_list))
        unique_context.append(unique_context_attr)

    return unique_context


def get_context_for_random_event(event, process_instances, process_instances_context):
    index = 0
    while True:
        process_instance_id = random.randrange(len(process_instances))
        if event in process_instances[process_instance_id]:
            return process_instances_context[process_instance_id][process_instances[process_instance_id].index(event)]
        if index == 25:
            return -1
        index = index + 1


def delete_encoders():
    folder = './encoder'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def f1_score(y_true, y_pred):
    """
    Computes the f1 score - performance indicator for the prediction accuracy.
    The F1 score is the harmonic mean of the precision and recall.
    The evaluation metric to be optimized during hyperparameter optimization.

    Parameters
    ----------
    y_true : Tensor, dtype=float32
        True labels.
    y_pred : Tensor, dtype=float32
        Predicted labels.

    Returns
    -------
    Tensor : dtype=float32
        The F1-Score.

    """

    def recall(y_true, y_pred):
        """
        Computes the recall (only a batch-wise average of recall), a metric for multi-label classification of
        how many relevant items are selected.

        Parameters
        ----------
        y_true : Tensor, dtype=float32
            True labels.
        y_pred : Tensor, dtype=float32
            Predicted labels.

        Returns
        -------
        Tensor : dtype=float32
            The recall.

        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Computes the precision (only a batch-wise average of precision), a metric for multi-label classification of
        how many selected items are relevant.

        Parameters
        ----------
        y_true : Tensor, dtype=float32
            True labels.
        y_pred : Tensor, dtype=float32
            Predicted labels.

        Returns
        -------
        Tensor : dtype=float32
            The precision.

        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    # in order to avoid division by 0, the constant epsilon is added
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))