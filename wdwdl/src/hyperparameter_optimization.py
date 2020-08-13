from sklearn.model_selection import ShuffleSplit
import numpy

args = ''
number_attributes = ''
time_steps = ''
is_autoencoder = ''

x_train = ''
y_train = ''
x_test = ''
y_test = ''


def create_data(input_data, output_data, preprocessor, arguments, is_ae=False):
    """
    Generates input data to train and test/evaluate a model during hyperparameter optimization (hpopt) with hyperas.
    """

    global args
    global number_attributes
    global time_steps
    global is_autoencoder

    global x_train
    global y_train
    global x_test
    global y_test

    args = arguments
    number_attributes = preprocessor.data_structure['encoding']['num_values_features'] - 2  # case + time
    time_steps = preprocessor.data_structure['meta']['max_length_process_instance']
    is_autoencoder = is_ae

    train_indices, test_indices = train_test_split_for_hyperparameter_optimization(input_data)
    x_train, y_train, x_test, y_test = retrieve_train_test_instances_and_output_datas(input_data, output_data, train_indices,
                                                                                test_indices, is_ae)


def train_test_split_for_hyperparameter_optimization(input_data):
    """
    Executes a split-validation for hyperparameter optimization (hpopt) with hyperas. Retrieves indices of training and
    test cases of the training set for hpopt.
    """

    shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test_hpopt, random_state=0)

    hpopt_train_indices = []
    hpopt_test_indices = []

    for train_indices, test_indices in shuffle_split.split(input_data):
        hpopt_train_indices.append(train_indices)
        hpopt_test_indices.append(test_indices)

    return train_indices, test_indices


def retrieve_train_test_instances_and_output_datas(input_data, output_data, train_indices, test_indices, is_ae):
    """
    Retrieves training and test process instances and output_datas of the training set for hyperparameter optimization (hpopt).
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