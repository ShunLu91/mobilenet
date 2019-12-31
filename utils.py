import os
import sys
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_transforms(args):
    assert args.dataset in ['cifar10', 'imagenet']
    if args.dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if args.resize:  # cifar10 or imagenet
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:  # cifar10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    return train_transform, valid_transform


def random_choice(path, layers, kernels, group=4):
    """
    Generate  rand_num & rand_conv for different path
    with same size but different range
    """
    choice = {}
    for i in range(layers):
        # num
        rand_num = list(np.random.randint(low=1, high=(1 + path), size=group))
        # conv
        rand_conv = []
        for j in rand_num:
            rand_conv.append(list(np.random.randint(low=0, high=kernels, size=j)))
        # expansion rate
        rand_rate = 0
        # rand_rate = random.choice((0, 1))
        # choice
        choice[i] = {'num': rand_num, 'conv': rand_conv, 'rate': rand_rate}
    return choice


def random_num_conv(path, layers, kernels):
    """
    Generate  rand_num & rand_conv for different path
    with same size but different range
    """
    rand_conv = []
    rand_num = []
    for i in range(layers):
        rand_num.append(list(np.random.randint(low=1, high=(1 + path), size=kernels)))
        _conv = []
        for j in rand_num[i]:
            _conv.append(list(np.random.randint(low=0, high=kernels, size=j)))
        rand_conv.append(_conv)
    return rand_num, rand_conv


def conv_to_num(rand_conv):
    rand_num = []
    for i in rand_conv:
        length = []
        for j in i:
            length.append(len(j))
        rand_num.append(length)
    return rand_num


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        if str(x.device) == 'cpu':
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)


def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("successfully save the txt")


def plot_hist(acc_list, min=0, max=101, interval=5, name='search'):
    plt.hist(acc_list, bins=max - min, range=(min, max), histtype='bar')
    plt.xticks(np.arange(min, max, interval))
    img_path = name + '.png'
    plt.savefig(img_path)
    plt.show()


def plot_hist_from_txt(txt_path, name):
    a = np.loadtxt(txt_path)
    print(list(a))
    plot_hist(list(a), name=name)


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))


def mixup_data(x, y, alpha=1.0, use_cuda=True, per_sample=False):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    if alpha > 0. and not per_sample:
        lam = torch.zeros(y.size()).fill_(np.random.beta(alpha, alpha)).cuda()
        mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index, :]
    elif alpha > 0.:
        lam = torch.Tensor(np.random.beta(alpha, alpha, size=y.size())).cuda()
        mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index, :]
    else:
        lam = torch.ones(y.size()).cuda()
        mixed_x = x

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_lam_idx(batch_size, alpha, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    return lam, index


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: criterion(pred, y_a, lam) + criterion(pred, y_b, 1 - lam)


########################################## Example of progress_bar ##########################################
# utils.progress_bar(epoch+1, step, len(train_data), 'Loss: %.3f | Acc: %.3f%%' % (train_loss / (step + 1), top1.avg))
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 150
last_time = time.time()
begin_time = last_time


def progress_bar(epoch, current, total, msg=None):
    TOTAL_BAR_LENGTH = 65.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Epoch: %d' % epoch)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Total: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':
    choice = random_choice(path=1, layers=8, kernels=1)
    print(choice)
    # index = []
    # for i in choice:
    #     index.append(choice[i]['conv'])
    #     index.append(choice[i]['rate'])
    # print(index)
