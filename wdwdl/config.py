import os
import argparse
import wdwdl.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--dnn_num_epochs', default=1, type=int)
    parser.add_argument('--dnn_num_epochs_auto_encoder', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)

    # pre-processing: min_max_norm, bin or onehot
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes
    # note: onehot encoding of the categorical attributes can lead to memory errors
    parser.add_argument('--encoding_cat', default="bin", type=str)  # for categorical attributes

    # all models
    parser.add_argument('--task', default="workaround_detection")
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # evaluation
    # note: cross validation is not implemented; only split validation
    parser.add_argument('--num_folds', default=3, type=int)  # 10
    parser.add_argument('--cross_validation', default=False, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.7, type=float)

    # hyperparameter optimization
    parser.add_argument('--hpopt', default=True, type=utils.str2bool)
    parser.add_argument('--hpopt_eval_runs', default=2, type=int)
    parser.add_argument('--split_rate_test_hpopt', default=0.2, type=float)
    parser.add_argument('--hpopt_LSTM', default=[50, 100, 150, 200], type=list)
    parser.add_argument('--hpopt_optimizers', default=['nadam', 'adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad'], type=list)
    parser.add_argument('--hpopt_activation', default=['tanh', 'linear', 'relu', 'elu'], type=list)
    parser.add_argument('--hpopt_kernel_initializer', default=['random_normal', 'random_uniform', 'glorot_normal',
                                                               'glorot_uniform', 'truncated_normal', 'zeros'], type=list)

    # data
    parser.add_argument('--data_set', default="bpi2013i_converted_selection.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")

    # gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
