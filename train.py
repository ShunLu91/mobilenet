import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
from tqdm import tqdm
from thop import profile
from torchvision import datasets
from utils import data_transforms
from model import MobileNetV2
from torchsummary import summary
import torch.backends.cudnn as cudnn

# warnings
# import warnings
# warnings.filterwarnings('ignore')


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    train_data = tqdm(train_data)
    train_data.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch+1, args.epochs, 'lr:', scheduler.get_lr()[0]))
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (step + 1)),
                   'top1': '%.6f' % top1.avg, 'top5': '%.6f' % top5.avg}
        train_data.set_postfix(postfix)


def validate(args, epoch, val_data, device, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
        print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
              % (epoch + 1, val_loss / (step + 1), val_top1.avg))
        return val_top1.avg


def main():
    # args & device
    args = config.get_args()
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    # MobileNetV2
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

    # dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012', 'train'), train_transform)
        val_data_set = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012', 'val'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

    if args.resume:
        resume_path = './snapshots/{}full_train_states.pt.tar'.format(args.exp_name)
        if os.path.isfile(resume_path):
            print("Loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.load_state_dict(checkpoint['supernet_state'])
            scheduler.laod_state_dict(checkpoint['scheduler_state'])
        else:
            raise ValueError("No checkpoint found at '{}'".format(resume_path))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train(args, epoch, train_loader, device, model, criterion, optimizer, scheduler)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_loader, device, model, criterion)
            utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name)


if __name__ == '__main__':
    main()
