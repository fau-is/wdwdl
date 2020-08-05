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
    print("Number training instances: %i" % len(data_set_train))
    print("Number test instances: %i" % len(data_set_test))

    # train
    best_model_id = trainer.train_nn_wa_classification(args, data_set_train, label_train, preprocessor)

    # test
    predictions = predictor.apply_wa_classification(args, data_set_test, preprocessor, best_model_id)
    utils.plot_confusion_matrix2(utils.arg_max(label_test).tolist(), predictions.tolist(), args)
    utils.calculate_and_print_output(utils.arg_max(label_test).tolist(), predictions.tolist())

    # predict
    data_set_pred = preprocessor.prepare_event_log_for_prediction()
    print("Number prediction instances: %i" % len(data_set_pred))
    predictions_ = predictor.apply_wa_classification(args, data_set_pred, preprocessor, best_model_id)
    print(predictor.get_prediction_frequency(predictions_))

    # delete encoders
    utils.delete_encoders()
