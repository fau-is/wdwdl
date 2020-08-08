import wdwdl.config as config
import wdwdl.predictor as predictor
import wdwdl.trainer as trainer
from wdwdl.src.preprocessing.preprocessor import Preprocessor
import wdwdl.src.utils.general as general
import wdwdl.src.utils.plot as plot
import wdwdl.src.utils.metric as metric

if __name__ == '__main__':

    args = config.load()

    # Delete encoders from previous experiments
    general.delete_encoders()

    # Init preprocessor
    preprocessor = Preprocessor(args)

    # Filter out noise and get no outliers
    no_outliers = preprocessor.clean_event_log(args)

    # Integrate workarounds
    data_set, label = preprocessor.add_workarounds_to_event_log(args, no_outliers)

    # Split data set
    data_set_train, data_set_test, label_train, label_test = general.train_test_ids_from_data_set(data_set, general.convert_label_to_categorical(label), 0.2)
    general.llprint("Number training instances: %i" % len(data_set_train))
    general.llprint("Number test instances: %i" % len(data_set_test))

    # Train model
    best_model_id = trainer.train_nn_wa_classification(args, data_set_train, label_train, preprocessor)

    # Test model
    predictions, prob_dist = predictor.apply_wa_classification(args, data_set_test, preprocessor, best_model_id)
    plot.confusion_matrix(general.arg_max(label_test).tolist(), predictions.tolist(), args)
    metric.calculate_and_print_output(general.arg_max(label_test).tolist(), label_test, predictions.tolist(), prob_dist)

    # Predict workarounds
    data_set_pred = preprocessor.prepare_event_log_for_prediction()
    general.llprint("Number prediction instances: %i\n" % len(data_set_pred))
    predictions_ = predictor.apply_wa_classification(args, data_set_pred, preprocessor, best_model_id)
    general.llprint(str(predictor.get_prediction_frequency(predictions_)))

    # Delete encoders
    general.delete_encoders()
