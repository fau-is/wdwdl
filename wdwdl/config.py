import os
import argparse
import wdwdl.src.utils.general as utils


def load():
    parser = argparse.ArgumentParser()

    # Data and pre-processing
    parser.add_argument('--data_set', default="bpi2012w_converted_selection.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes min-max norm
    parser.add_argument('--encoding_cat', default="bin", type=str)  # for categorical attributes binary encoding;
    # note one-hot encoding of the categorical attributes can lead to memory errors

    # Training
    parser.add_argument('--task', default="workaround_detection")
    parser.add_argument('--dnn_num_epochs', default=1, type=int)
    parser.add_argument('--dnn_num_epochs_auto_encoder', default=1, type=int)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # Evaluation
    parser.add_argument('--seed', default=True, type=utils.str2bool)
    parser.add_argument('--seed_val', default=1377, type=int)

    # Note cross validation is not implemented; only split validation
    parser.add_argument('--num_folds', default=3, type=int)  # 10
    parser.add_argument('--cross_validation', default=False, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.7, type=float)

    # Hyper-parameter optimization
    parser.add_argument('--hpopt', default=True, type=utils.str2bool)
    parser.add_argument('--hpopt_eval_runs', default=1, type=int)
    parser.add_argument('--split_rate_test_hpopt', default=0.1, type=float)  # size of validation set in hpo

    # parser.add_argument('--hpopt_optimizers', default=['nadam', 'adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad'], type=list)
    parser.add_argument('--hpopt_optimizers', default=['nadam', 'adam', 'rmsprop'], type=list)
    parser.add_argument('--hpopt_activation', default=['linear', 'tanh', 'relu', 'elu'], type=list)
    parser.add_argument('--hpopt_filters', default=[128, 64, 32], type=list)
    parser.add_argument('--hpopt_kernels_size', default=[32, 16, 8], type=list)
    # Stride = moving distance of filter in pixels/positions to right, often used values are 1 or 2
    # We want to grasp all potential workarounds in event log data, thus we set stride = 1
    parser.add_argument('--hpopt_strides', default=[1], type=list)
    parser.add_argument('--hpopt_padding', default=["same", "causal"], type=list)  # not 'valid'
    parser.add_argument('--hpopt_kernel_initializer', default=['glorot_normal', 'glorot_uniform'], type=list)
    parser.add_argument('--hpopt_units', default=[50, 100, 150], type=list)

    # Gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
