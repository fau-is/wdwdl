import wdwdl.config as config
import wdwdl.predictor as test_dnn
import wdwdl.trainer as train_dnn
from wdwdl.preprocessor import Preprocessor
import wdwdl.utils as utils

if __name__ == '__main__':

    args = config.load()
    # init preprocessor
    preprocessor = Preprocessor(args)
    # filter out noise and get no outliers
    no_outliers = preprocessor.clean_event_log(args)
    # integrate workaround forms
    data, label = preprocessor.add_workarounds_to_event_log(args, no_outliers)


    # preprocessor.split_event_log(args)

    # output = utils.load_output()
    utils.clear_measurement_file(args)



    """
    if args.cross_validation:

        for iteration_cross_validation in range(0, args.num_folds):
            preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation
            output["training_time_seconds"].append(train_dnn.train(args, preprocessor))
            test_dnn.test(args, preprocessor)
            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, iteration_cross_validation)
            utils.write_output(args, output, iteration_cross_validation)

        utils.print_output(args, output, iteration_cross_validation + 1)
        utils.write_output(args, output, iteration_cross_validation + 1)

    # split validation
    else:
        train_dnn(args, preprocessor)
        output["training_time_seconds"].append(train_dnn.train(args, preprocessor))
        test_dnn.test(args, preprocessor)
        output = utils.get_output(args, preprocessor, output)
        utils.print_output(args, output, -1)
        utils.write_output(args, output, -1)
    """