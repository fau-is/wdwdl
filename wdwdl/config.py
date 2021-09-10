import os
import argparse
import wdwdl.src.utils.general as general


def load():
    parser = argparse.ArgumentParser()

    # Data and pre-processing
    parser.add_argument('--data_set', default="bpi2012w_converted_sample.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # for numerical attributes min-max norm
    parser.add_argument('--encoding_cat', default="bin", type=str)  # for categorical attributes binary encoding
    parser.add_argument('--verbose', default=False, type=general.str2bool)

    # Training
    parser.add_argument('--task', default="workaround_detection")
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_num_epochs_auto_encoder', default=100, type=int)
    parser.add_argument('--batch_size_train', default=128, type=int)  # higher batch size lead to sharp minima
    parser.add_argument('--batch_size_test', default=1, type=int)

    # Evaluation
    parser.add_argument('--seed', default=True, type=general.str2bool)
    parser.add_argument('--seed_val', default=1377, type=int)
    parser.add_argument('--shuffle', default=True, type=int)
    parser.add_argument('--val_model_split', default=0.1, type=float)  # i.e., 10% of training data is used for validation
    parser.add_argument('--train_split', default=0.2, type=float)  # i.e., 20% of training data is used for training
    parser.add_argument('--pred_split', default=0.1, type=float)  # i.e., 10% of original data is used for prediction
    parser.add_argument('--split_rate_test_hpo', default=0.9, type=float)  # size of training set in hpo

    parser.add_argument('--remove_noise', default=False, type=int)
    parser.add_argument('--remove_noise_threshold', default=50, type=int)  # [50, 55, ..., 95]
    parser.add_argument('--remove_noise_augmentation', default=True, type=general.str2bool)

    parser.add_argument('--hpo', default=True, type=general.str2bool)  # Hyper-parameter optimization dnn
    parser.add_argument('--hpo_ae', default=True, type=general.str2bool)  # Hyper-parameter optimization autoencoder

    parser.add_argument('--hpo_eval_runs', default=10, type=int)
    parser.add_argument('--hpo_optimizer', default=['adam', 'nadam'], type=list)  # general.ams_grad() hyper-parameters
    parser.add_argument('--hpo_activation', default=['linear', 'tanh', 'relu'], type=list)
    parser.add_argument('--hpo_filters', default=[256, 128, 64, 32], type=list)  # 256, 128, 64, 32
    parser.add_argument('--hpo_kernels_size', default=[64, 32, 16, 8], type=list)  # 64, 32, 16, 8
    parser.add_argument('--hpo_strides', default=[1], type=list)  # Stride = moving distance of filter in pixels/positions to right, often used values are 1 or 2; # We want to grasp all potential workarounds in event log data, thus we set stride = 1
    parser.add_argument('--hpo_padding', default=["same", "causal"], type=list)  # not 'valid'
    parser.add_argument('--hpo_kernel_initializer', default=['glorot_normal', 'glorot_uniform'], type=list)
    parser.add_argument('--hpo_units', default=[50, 100, 150], type=list)
    parser.add_argument('--hpo_conv_layers', default=[3, 4, 5], type=list)
    parser.add_argument('--hpo_ae_layers', default=[1, 3, 5], type=list)
    parser.add_argument('--hpo_ae_enc_dim', default=[256, 128, 64], type=list)

    # Gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
