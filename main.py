import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
from thop import profile
from torchvision import datasets
from utils import data_transforms
from mobilenetv2 import MobileNetV2
from torchsummary import summary


def main():
    # args & device
    args = config.get_args()
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # # dataset
    # train_transform, valid_transform = data_transforms(args)
    # if args.dataset == 'cifar10':
    #     trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
    #                                             download=True, transform=train_transform)
    #     train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
    #                                                shuffle=True, pin_memory=True, num_workers=8)
    #     valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
    #                                           download=True, transform=valid_transform)
    #     val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
    #                                              shuffle=False, pin_memory=True, num_workers=8)
    # elif args.dataset == 'imagenet':
    #     train_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'train'), train_transform)
    #     val_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'valid'), valid_transform)
    #     train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
    #                                                num_workers=8, pin_memory=True, sampler=None)
    #     val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
    #                                              num_workers=8, pin_memory=True)

    # SinglePath_OneShot
    model = MobileNetV2()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=1e-8, last_epoch=-1)

    # flops & params & structure
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),) if args.dataset == 'cifar10'
                            else (torch.randn(1, 3, 224, 224),), verbose=False)
    # print(model)
    print('Params: %.2fM, Flops:%.2fM' % ((params / 1e6), (flops / 1e6)))
    model = model.to(device)
    summary(model, (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224))







if __name__ == '__main__':
    main()