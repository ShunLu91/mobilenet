import argparse


def get_args():
    parser = argparse.ArgumentParser("Multi_Path_One-Shot")
    parser.add_argument('-e', '--exp_name', type=str, required=True, help='search model name')
    parser.add_argument('--path', type=int, default=1, help='num of selected paths')
    parser.add_argument('--choice_index', type=int, default=0, help='num of selected paths')
    parser.add_argument('--shadow_bn', action='store_true', help='shadow bn or not, default: False')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--search_num', type=int, default=1000, help='num of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
    parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop_path_prob')
    parser.add_argument('--use_se', action='store_true', default=False, help='use SqueezeExcite or not')
    parser.add_argument('--resume', action='store_true', default=False, help='resume')
    # ******************************* dataset *******************************#
    parser.add_argument('--dataset', type=str, default='imagenet', help='[cifar10, imagenet]')
    parser.add_argument('--data_dir', type=str, default='/dataset/imagenet', help='dataset dir')
    parser.add_argument('--classes', type=int, default=1000, help='classes of the dataset')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
    parser.add_argument('--resize', action='store_true', default=False, help='use resize')
    parser.add_argument('--alpha', default=0.2, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    args = parser.parse_args()
    print(args)
    return args
