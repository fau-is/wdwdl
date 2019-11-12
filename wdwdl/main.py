import wdwdl.config as config
import wdwdl.predictor as predictor
import wdwdl.trainer as trainer
from wdwdl.preprocessor import Preprocessor
import wdwdl.utils as utils
import keras

if __name__ == '__main__':

    args = config.load()

    # init preprocessor
    preprocessor = Preprocessor(args)

    # filter out noise and get no outliers
    no_outliers = preprocessor.clean_event_log(args)

    # integrate workaround forms
    data_set, label = preprocessor.add_workarounds_to_event_log(args, no_outliers)

    # split data set
    data_set_train, data_set_test, label_train, label_test = preprocessor.split_validation(data_set, utils.convert_label_to_categorical(label), 0.2)

    # train
    trainer.train_nn_wa_classification(args, data_set_train, label_train)

    # test
    predictions = predictor.apply_wa_classification(args, data_set_test)
    utils.plot_confusion_matrix(utils.arg_max(label_test).tolist(), predictions.tolist())
    utils.calculate_and_print_output(utils.arg_max(label_test).tolist(), predictions.tolist())

    # predict
    # predictions_ = predictor.apply_wa_classification(args, data_set_test)


