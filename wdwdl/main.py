import wdwdl.config as config
import wdwdl.predictor as tester
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
    # convert label
    label = keras.utils.np_utils.to_categorical(label)
    # split data set
    data_set_train, data_set_test, label_train, label_test = preprocessor.split_validation(data_set, label)
    # train
    trainer.train_nn_wa_classification(args, data_set_train, label_train)
    # test
    
    print(0)
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