import argparse
import pickle
import sys
import numpy
import csv
import sklearn
import arrow
import os
import keras
import matplotlib.pyplot as pyplot

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

class_names = ["No Workaround",
               "injured_responsibility",
               "manipulated_data",
               "repeated_activity",
               "substituted_activity",
               "interchanged_activity",
               "bypassed_activity",
               "added_activity",
               "added_activity"
               ]


def arg_max(list_):
    return numpy.argmax(list_, axis=1)


def convert_label_to_categorical(label):
    return keras.utils.np_utils.to_categorical(label)


def load(path):
    return pickle.load(open(path, 'rb'))


def load_output():
    return output


def avg(numbers):
    if len(numbers) == 0:
        return sum(numbers)

    return sum(numbers) / len(numbers)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = numpy.zeros(int(size), dtype=numpy.float32)
    vec[int(index)] = 1.0
    return vec


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open('./results/output_%s.csv' % (args.data_set[:-4]), "w").close()


def get_output(args, preprocessor, _output):
    prefix = 0
    prefix_all_enabled = 1

    predicted_label = list()
    ground_truth_label = list()

    if not args.cross_validation:
        result_dir_fold = \
            args.result_dir + \
            args.data_set.split(".csv")[0] + \
            "__" + args.task + \
            "_0.csv"
    else:
        result_dir_fold = \
            args.result_dir + \
            args.data_set.split(".csv")[0] + \
            "__" + args.task + \
            "_%d" % preprocessor.data_structure['support']['iteration_cross_validation'] + ".csv"

    with open(result_dir_fold, 'r') as result_file_fold:
        result_reader = csv.reader(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))
    _output["auc_roc_values"].append(multi_class_roc_auc_score(ground_truth_label, predicted_label))
    # we use the average precision at different threshold values as auc of the pr-curve
    # and not the auc-pr-curve with the trapezoidal rule / linear interpolation, because it could be too optimistic
    _output["auc_prc_values"].append(multi_class_prc_auc_score(ground_truth_label, predicted_label))

    return _output


def plot_confusion_matrix(label_ground_truth, label_prediction,
                          classes=class_names,
                          normalize=False,
                          title=None,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    numpy.set_printoptions(precision=2)

    label_ground_truth = numpy.array(label_ground_truth)
    label_prediction = numpy.array(label_prediction)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = sklearn.metrics.confusion_matrix(label_ground_truth, label_prediction)
    # todo: map to class_names -> classes[]
    classes = sklearn.utils.multiclass.unique_labels(label_ground_truth, label_prediction)
    classes_ = []
    for element in classes:
        classes_.append(class_names[element])
    classes = classes_
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


def print_output(args, _output, index_fold):
    if args.cross_validation and index_fold < args.num_folds:
        llprint("\nAccuracy of fold %i: %f\n" % (index_fold, _output["accuracy_values"][index_fold]))
        llprint("Precision of fold %i: %f\n" % (index_fold, _output["precision_values"][index_fold]))
        llprint("Recall of fold %i: %f\n" % (index_fold, _output["recall_values"][index_fold]))
        llprint("F1-score of fold %i: %f\n" % (index_fold, _output["f1_values"][index_fold]))
        llprint("Auc-roc of fold %i: %f\n" % (index_fold, _output["auc_roc_values"][index_fold]))
        llprint("Auc-prc of fold %i: %f\n" % (index_fold, _output["auc_prc_values"][index_fold]))
        llprint("Training time of fold %i: %f seconds\n\n" % (index_fold, _output["training_time_seconds"][index_fold]))

    else:
        llprint("\nAccuracy avg: %f\n" % (avg(_output["accuracy_values"])))
        llprint("Precision avg: %f\n" % (avg(_output["precision_values"])))
        llprint("Recall avg: %f\n" % (avg(_output["recall_values"])))
        llprint("F1-score avg: %f\n" % (avg(_output["f1_values"])))
        llprint("Auc-roc avg: %f\n" % (avg(_output["auc_roc_values"])))
        llprint("Auc-prc avg: %f\n" % (avg(_output["auc_prc_values"])))
        llprint("Training time avg: %f seconds" % (avg(_output["training_time_seconds"])))


def get_mode(index_fold, args):
    """ Gets the mode - split, fold or avg. """

    if index_fold == -1:
        return "split-%s" % args.split_rate_test
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):
    """ If fold < max number of folds in cross validation than use a specific value, else avg works. In addition,
    this holds for split. """

    if _mode != "split-%s" % args.split_rate_test and _mode != "avg":
        return _output[measure][_index_fold]
    else:
        return avg(_output[measure])


def write_output(args, _output, index_fold):
    """ Writes the output. """

    with open('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task), mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')

        # if file is empty
        if os.stat('./results/output_%s_%s.csv' % (args.data_set[:-4], args.task)).st_size == 0:
            writer.writerow(
                ["experiment[ds-cat_enc-dl_arch]", "mode", "validation", "accuracy", "precision", "recall", "f1-score",
                 "auc-roc", "auc-prc", "training-time",
                 "time-stamp"])
        writer.writerow([
            "%s-%s-%s" % (args.data_set[:-4], args.encoding_cat, args.dnn_architecture),  # experiment
            get_mode(index_fold, args),  # mode
            "cross-validation" if args.cross_validation else "split-validation",  # validation
            get_output_value(get_mode(index_fold, args), index_fold, _output, "accuracy_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_roc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_prc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "training_time_seconds", args),
            arrow.now()
        ])


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

