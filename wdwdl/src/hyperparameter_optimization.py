import wdwdl.src.utils.general as general
import numpy

args = ''
number_attributes = ''
time_steps = ''
is_autoencoder = ''

x_train = ''
y_train = ''
x_test = ''
y_test = ''


def create_data(input_data, output_data, preprocessor, args_, is_ae=False):
    """
    Generates data to train and test/evaluate a model during hyper-parameter optimization (hpo) with Optuna.
    """

    global args
    global number_attributes
    global time_steps
    global is_autoencoder

    global x_train
    global y_train
    global x_test
    global y_test

    args = args_
    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    is_autoencoder = is_ae

    train_indices, test_indices = train_test_split_for_hpo(args, input_data)
    x_train, y_train, x_test, y_test = retrieve_train_test_instances_and_output_datas(input_data, output_data,
                                                                                      train_indices,
                                                                                      test_indices, is_ae)


def train_test_split_for_hpo(args, input_data):
    """
    Executes a split-validation and retrieves indices of training and test data for hpo.
    :param input_data:
    :type args: object
    :return: train_indices, test_indices
    """

    return general.train_test_ids_from_data_set(args, input_data)


def retrieve_train_test_instances_and_output_datas(input_data, output_data, train_indices, test_indices, is_ae):
    """
    Retrieves training and test instances from indices for hpo.
    """
    global time_steps
    global number_attributes

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for idx in train_indices:
        x_train.append(input_data[idx])
        y_train.append(output_data[idx])

    for idx in test_indices:
        x_test.append(input_data[idx])
        y_test.append(output_data[idx])

    x_train = numpy.asarray(x_train)
    x_test = numpy.asarray(x_test)
    y_train = numpy.asarray(y_train)
    y_test = numpy.asarray(y_test)

    if not is_ae:
        x_train = x_train.reshape((x_train.shape[0], time_steps, number_attributes))
        x_test = x_test.reshape((x_test.shape[0], time_steps, number_attributes))

    return x_train, y_train, x_test, y_test
