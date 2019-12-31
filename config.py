import time
import utils
import torch
import argparse
import torchvision


def get_args():
    parser = argparse.ArgumentParser("Multi_Path_One-Shot")
    parser.add_argument('--exp_name', type=str, required=True, help='search model name')
    parser.add_argument('--path', type=int, default=1, help='num of selected paths')
    parser.add_argument('--choice_index', type=int, default=0, help='num of selected paths')
    parser.add_argument('--shadow_bn', action='store_true', help='shadow bn or not, default: False')
    parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='dataset dir')
    parser.add_argument('--classes', type=int, default=10, help='num of MB_layers')
    parser.add_argument('--layers', type=int, default=12, help='num of MB_layers')
    parser.add_argument('--group', type=int, default=4, help='num of groups')
    parser.add_argument('--kernels', type=int, default=4, help='num of kernels')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--search_num', type=int, default=1000, help='num of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
    parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop_path_prob')
    parser.add_argument('--use_se', action='store_true', default=False, help='use SqueezeExcite or not')
    # ******************************* dataset *******************************#
    parser.add_argument('--dataset', type=str, default='imagenet', help='[cifar10, imagenet]')
    parser.add_argument('--cutouexp_namet', action='store_false', help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
    parser.add_argument('--resize', action='store_true', default=False, help='use resize')
    parser.add_argument('--alpha', default=0.2, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # ******************************* ea_search *******************************#
    # parser.add_argument('--log-dir', type=str, default='snapshots')
    # parser.add_argument('--input_shape', type=int, default=(1, 3, 32, 32), help='input_shape')
    # parser.add_argument('--max-epochs', type=int, default=20)
    # parser.add_argument('--select-num', type=int, default=10)
    # parser.add_argument('--population-num', type=int, default=50)
    # parser.add_argument('--m_prob', type=float, default=0.1)
    # parser.add_argument('--crossover-num', type=int, default=25)
    # parser.add_argument('--mutation-num', type=int, default=25)
    # parser.add_argument('--flops-limit', type=float, default=600)
    # parser.add_argument('--max-train-iters', type=int, default=200)
    # parser.add_argument('--max-test-iters', type=int, default=40)
    # parser.add_argument('--train-batch-size', type=int, default=128)
    # parser.add_argument('--test-batch-size', type=int, default=200)
    arguments = parser.parse_args()
    print(arguments)
    return arguments