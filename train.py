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
from model import MobileNetV2
from torchsummary import summary
import torch.backends.cudnn as cudnn
from utils import data_transforms, set_seed, eta_time

# warnings
import warnings
warnings.filterwarnings('ignore')


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
    val_top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)
        print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
              % (epoch + 1, val_loss / (step + 1), val_top1.avg))
        return val_top1.avg, val_top5.avg, val_loss / (step + 1)


def main():
    # prepare dir
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')

    # args
    args = config.get_args()
    # seed
    set_seed(args.seed)
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
    print('Params: %.5fM, Flops:%.5fM' % ((params / 1e6), (flops / 1e6)))
    model = model.to(device)
    summary(model, (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224))

    # dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
                                                download=False, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                              download=False, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'imagenet':
        train_data = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), train_transform)
        val_data = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=16, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=16, pin_memory=True)

    if args.resume:
        resume_path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
        if os.path.isfile(resume_path):
            print("Loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.load_state_dict(checkpoint['supernet_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        else:
            raise ValueError("No checkpoint found at '{}'".format(resume_path))
    else:
        start_epoch = 0

    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()

        # train
        train(args, epoch, train_loader, device, model, criterion, optimizer, scheduler)
        scheduler.step()

        # validate
        val_top1, val_top5, val_loss = validate(args, epoch, val_loader, device, model, criterion)
        elapse = time.time() - t1
        h, m, s = eta_time(elapse, args.epochs - epoch - 1)

        # save best model
        if val_top1 > best_acc:
            best_acc = val_top1
            # save the states of this epoch
            state = {
                'epoch': epoch,
                'args': args,
                'optimizer_state': optimizer.state_dict(),
                'supernet_state': model.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
            torch.save(state, path)
            # print('\n best val acc: {:.6}'.format(best_acc))
        print('\nval: loss={:.6}, top1={:.6}, top5={:.6}, best={:.6}, elapse={:.0f}s, eta={:.0f}h {:.0f}m {:.0f}s\n'
              .format(val_loss, val_top1, val_top5, best_acc, elapse, h, m, s))
    print('Best Top1 Acc: {:.6}'.format(best_acc))


if __name__ == '__main__':
    main()
