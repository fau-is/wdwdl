import os
import argparse
import wdwdl.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_num_epochs_auto', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)

    # pre-processing (min_max_norm, int, bin, onehot or hash)
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # for categorical attributes
    parser.add_argument('--num_hash_output', default=2, type=int)  # number of output columns of hash encoding

    # all models
    parser.add_argument('--task', default="workaround_detection")
    parser.add_argument('--learning_rate', default=0.002, type=float)  # dnc 0.0001 #dnn 0.002

    # evaluation
    parser.add_argument('--num_folds', default=3, type=int)  # 10
    parser.add_argument('--cross_validation', default=False, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.7, type=float)  # only if cross validation is deactivated
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set', default="bpi2012w_converted_selection.csv")  #bpi2012w_converted_sample.csv")
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
