import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
from thop import profile
from torchvision import datasets
from model import MobileNetV2
from tqdm import tqdm
import horovod.torch as hvd
from torchsummary import summary
import torch.backends.cudnn as cudnn
from utils import data_transforms, set_seed, elapse_time, eta_time

# args
args = config.get_args()
# warnings
import warnings
warnings.filterwarnings('ignore')
# SummaryWriter
from tensorboardX import SummaryWriter
train_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Train')
val_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Val')


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for step, (inputs, targets) in enumerate(tqdm(train_data)):
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
    train_writer.add_scalar('Loss', train_loss / (step + 1), epoch)
    train_writer.add_scalar('Acc', top1.avg, epoch)

    print('Epoch:{:0>4d}/{:0>4d}, lr:{:.5f}, loss:{:.6f}, top1:{:.6f}, top5:{:.6f}'
          .format(epoch + 1, args.epochs, scheduler.get_lr()[0], train_loss / (step + 1), top1.avg, top5.avg))


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
        print('[Val_Accuracy epoch:{:0>4d}] val_loss:{:.6f}, val_top1:{:.6f}, val_top5:{:.6f}'
              .format(epoch + 1, val_loss / (step + 1), val_top1.avg, val_top5.avg))
    val_writer.add_scalar('Loss', val_loss / (step + 1), epoch)
    val_writer.add_scalar('Acc', val_top1.avg, epoch)

    return val_top1.avg, val_top5.avg, val_loss / (step + 1)


def main():
    # prepare dir
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')
    # seed
    set_seed(args.seed)
    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        # torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    # 1. 初始化horovod
    hvd.init()
    # 2. 给当前进程分配对应的gpu，local_rank()返回的是当前是第几个进程
    torch.cuda.set_device(hvd.local_rank())

    # MobileNetV2
    model = MobileNetV2().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # parrallel
    # model = nn.DataParallel(model)
    # criterion = nn.DataParallel(criterion)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # 5. 初始化的时候广播参数，这个是为了在一开始的时候同步各个gpu之间的参数
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=1e-8, last_epoch=-1)

    # # flops & params & structure
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),) if args.dataset == 'cifar10'
    # else (torch.randn(1, 3, 224, 224),), verbose=False)
    # # print(model)
    # print('Params: %.5fM, Flops:%.5fM' % ((params / 1e6), (flops / 1e6)))
    # model = model.to(device)
    # summary(model, (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224))

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
        # 3. 用DistributedSampler给各个worker分数据
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=32, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=32, pin_memory=True)

    # resume
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
        elapse = time.time() - t1
        h0, m0, s0 = elapse_time(elapse)
        h1, m1, s1 = eta_time(elapse, args.epochs - epoch - 1)
        # validate
        if (epoch+1) % args.val_interval == 0 :
            val_top1, val_top5, val_loss = validate(args, epoch, val_loader, device, model, criterion)
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
        print('val_best={:.6f}, elapse={:.0f}h{:.0f}m{:.0f}s, eta={:.0f}h{:.0f}m{:.0f}s\n'
              .format(best_acc, h0, m0, s0, h1, m1, s1))
    print('Best Val Top1 Acc: {:.6}'.format(best_acc))


if __name__ == '__main__':
    main()
