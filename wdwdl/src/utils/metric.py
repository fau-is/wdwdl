import numpy as np
import sklearn
import wdwdl.src.utils.general as general
import warnings
import tensorflow.keras.backend as K


def calculate_and_print_output(label_ground_truth, label_prediction):
    """
    This function calculate and prints the measures.
    :param label_ground_truth:
    :param label_prediction:
    :return:
    """

    np.set_printoptions(precision=3)
    warnings.filterwarnings("ignore")

    label_ground_truth = np.array(label_ground_truth)
    label_prediction = np.array(label_prediction)

    general.llprint("\nAccuracy: %f\n" % sklearn.metrics.accuracy_score(label_ground_truth, label_prediction))
    general.llprint(
        "Precision: %f\n" % sklearn.metrics.precision_score(label_ground_truth, label_prediction, average='macro'))
    general.llprint(
        "Recall: %f\n" % sklearn.metrics.recall_score(label_ground_truth, label_prediction, average='macro'))
    general.llprint("F1-score: %f\n" % sklearn.metrics.f1_score(label_ground_truth, label_prediction, average='macro'))
    general.llprint("Auc-roc: %f\n" % multi_class_roc_auc_score(label_ground_truth, label_prediction))
    # we use the average precision at different threshold values as auc of the pr-curve
    # and not the auc-pr-curve with the trapezoidal rule / linear interpolation, because it could be too optimistic
    general.llprint("Auc-prc: %f\n\n" % multi_class_prc_auc_score(label_ground_truth, label_prediction))


def multi_class_prc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    """
    Calculate prc_auc_score
    :param ground_truth_label:
    :param predicted_label:
    :param average:
    :return:
    """

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.average_precision_score(ground_truth_label, predicted_label, average=average)


def multi_class_roc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    """
    Calculate roc_auc_score
    :param ground_truth_label:
    :param predicted_label:
    :param average:
    :return:
    """

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.roc_auc_score(ground_truth_label, predicted_label, average=average)


def f1_score(y_true, y_pred):
    """
    Computes the f1 score - performance indicator for the prediction accuracy.
    The F1 score is the harmonic mean of the precision and recall.
    The evaluation metric to be optimized during hyper-parameter optimization.
    :param y_true: Tensor, dtype=float32
    :param y_pred: Tensor, dtype=float32
    :return: f1_score, Tensor : dtype=float32
    """

    def recall(y_true, y_pred):
        """
        Computes the recall (only a batch-wise average of recall), a metric for multi-label classification of
        how many relevant items are selected.
        :param y_true: Tensor, dtype=float32
        :param y_pred: Tensor, dtype=float32
        :return: recall, Tensor : dtype=float32
        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Computes the precision (only a batch-wise average of precision), a metric for multi-label classification of
        how many selected items are relevant.
        :param y_true: Tensor, dtype=float32
        :param y_pred: Tensor, dtype=float32
        :return: precision, Tensor : dtype=float32
        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    # To avoid division by 0, the constant epsilon is added
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
