import wdwdl.config as config
import wdwdl.src.predictor as predictor
import wdwdl.src.trainer as trainer
from wdwdl.src.preprocessing.preprocessor import Preprocessor
import wdwdl.src.utils.general as general
import wdwdl.src.utils.plot as plot
import wdwdl.src.utils.metric as metric


def run_experiment(args, ex_id):
    # Init
    results = metric.metrics()
    general.delete_encoders()
    # general.clear_files(args)

    # For reproducible results
    if args.seed:
        general.set_seed(args)

    # Init preprocessor
    preprocessor = Preprocessor(args, ex_id)

    # Filter out noise and get no outliers
    no_outliers = preprocessor.clean_event_log(args, preprocessor)

    # Integrate workarounds
    data_set, label = preprocessor.add_workarounds_to_event_log(args, no_outliers)

    # Split data set
    data_set_train, data_set_test, label_train, label_test = general.train_test_sets_from_data_set(args, data_set,
                                                                                                   general.convert_label_to_categorical(
                                                                                                       label),
                                                                                                   args.train_split)
    general.llprint("Number training instances: %i\n" % len(data_set_train))
    general.llprint("Number test instances: %i\n" % len(data_set_test))

    # Train model
    best_model_id = trainer.train_nn_wa_classification(args, data_set_train, label_train, preprocessor)

    # Test model
    predictions, prob_dist = predictor.apply_wa_classification(args, data_set_test, preprocessor, best_model_id)
    plot.confusion_matrix(general.arg_max(label_test).tolist(), predictions.tolist(), args)
    metric.calculate_and_print_output(args, general.arg_max(label_test).tolist(), label_test, predictions.tolist(),
                                      prob_dist, results)

    # Predict workarounds
    data_set_pred = preprocessor.prepare_event_log_for_prediction()
    general.llprint("Number prediction instances: %i\n" % len(data_set_pred))
    predictions_, prob_dist = predictor.apply_wa_classification(args, data_set_pred, preprocessor, best_model_id)
    prediction_frequencies = str(predictor.get_prediction_frequency(predictions_))
    general.llprint(prediction_frequencies)
    general.add_to_file(args, "predictions", prediction_frequencies)

    # Delete encoders
    general.delete_encoders()


def set_params_experiment(args, params, ex_id):
    args.data_set = params[ex_id]["data_set"]
    args.remove_noise = params[ex_id]["remove_noise"]
    args.remove_noise_threshold = params[ex_id]["remove_noise_threshold"]

    return args


if __name__ == '__main__':

    args = config.load()

    params = {
        # 0: {"data_set": "bpi2012w_converted_selection.csv", "remove_noise": True, "remove_noise_threshold": 50},
        # 1: {"data_set": "bpi2013i_converted_selection.csv", "remove_noise": True, "remove_noise_threshold": 50},
        # 2: {"data_set": "bpi2019_converted_selection.csv", "remove_noise": True, "remove_noise_threshold": 70},
        # 3: {"data_set": "bpi2020_converted_selection.csv", "remove_noise": True, "remove_noise_threshold": 70}

        0: {"data_set": "bpi2020_converted_selection.csv", "remove_noise": True, "remove_noise_threshold": 70}

    }

    for ex_id in range(0, len(params.keys())):
        # Set params for experiment
        args = set_params_experiment(args, params, ex_id)

        # Run experiment
        run_experiment(args, ex_id)
