import torch
import argparse
import torch.cuda
import random
import torch.nn as nn
from torch.nn import *
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

def calcshape(model):
    s = [(3, 32, 32)]
    l = (len(model) - 6) // 3
    for i in range(l):
        c, h, w = s[i]
        c, (kx, ky), (px, py) = model[3 * i].out_channels, model[3 * i].kernel_size, model[3 * i].padding
        h, w = h + 2 * px - kx + 1, w + 2 * py - ky + 1
        (kx, ky), st = model[3 * i + 2].kernel_size, model[3 * i + 2].stride
        h, w = (h - kx) // st + 1, (w - ky) // st + 1
        s.append((c, h, w))
    return s

def modify(model, n_NM):
    m = model.copy()
    for i in range(n_NM):
        s = calcshape(m)
        flg = True
        oloop = 0
        while flg and oloop < 10000:
            type = random.randint(0, 1)
            oloop += 1
            if (type == 0):
                p = random.randint(0, len(s) - 1)
                if (p < len(s) - 1):
                    c1, h1, w1 = s[p]
                    c2, h2, w2 = s[p + 1]
                    if (h1 < h2 - 4 or w1 < w2 - 4):
                        continue
                    st = random.randint(1, min((h1 + 3) // (h2 - 1), (w1 + 3) // (w2 - 1)))
                    l1, r1 = h1 + 2 - st * h2, h1 + 1 - st * (h2 - 1)
                    l2, r2 = w1 + 2 - st * w2, w1 + 1 - st * (w2 - 1)
                    while True:
                        x, kx, px = random.randint(2, r1 + 6), random.randint(2, r1 + 6), random.randint(0, 3)
                        if (l1 <= x + kx - 2 * px and x + kx - 2 * px <= r1):
                            break
                    while True:
                        y, ky, py = random.randint(2, r2 + 6), random.randint(2, r2 + 6), random.randint(0, 3)
                        if (l2 <= y + ky - 2 * py and y + ky - 2 * py <= r2):
                            break
                    del(m[3 * p])
                    del(m[3 * p])
                    del(m[3 * p])
                    m.insert(3 * p, nn.MaxPool2d((kx, ky), st))
                    m.insert(3 * p, nn.ReLU())
                    m.insert(3 * p, nn.Conv2d(c1, c2, (x, y), padding = (px, py)))
                    flg = False
                else:
                    c1, h1, w1 = s[p]
                    c2 = random.randint(3, c1 + 3)
                    iloop = 0;
                    while True and iloop < 100000:
                        x, y = random.randint(1, h1 // 2), random.randint(1, w1 // 2)
                        h2, w2 = h1 - x + 1, w1 - y + 1
                        iloop += 1
                        if (h2 > 2 and w2 > 2):
                            break
                    if (iloop == 100000):
                        continue
                    del(m[3 * p])
                    del(m[3 * p])
                    st = random.randint(1, min(h2 - 2, w2 - 2))
                    kx, ky = random.randint(2, h2 - st), random.randint(2, w2 - st)
                    h2, w2 = (h2 - kx) // st + 1, (w2 - ky) // st + 1
                    m.insert(3 * p, nn.Linear(c2 * h2 * w2, 120))
                    m.insert(3 * p, Reshape(-1, c2 * h2 * w2))
                    m.insert(3 * p, nn.MaxPool2d((kx, ky), st))
                    m.insert(3 * p, nn.ReLU())
                    m.insert(3 * p, nn.Conv2d(c1, c2, (x, y)))
                    flg = False
            else:
                p = random.randint(0, len(s) - 2)
                c1, h1, w1 = s[p]
                c2, h2, w2 = s[p + 1]
                if (h1 < h2 - 8 or w1 < w2 - 8):
                    continue
                c0 = random.randint(max(min(c1, c2) - 4, 2), max(c1, c2) + 4)
                h0 = random.randint(max(h2 - 4, 2), h1 + 4)
                w0 = random.randint(max(w2 - 4, 2), w1 + 4)
                st = random.randint(1, min((h0 + 3) // (h2 - 1), (w0 + 3) // (w2 - 1)))
                l1, r1 = h0 + 2 - st * h2, h0 + 1 - st * (h2 - 1)
                l2, r2 = w0 + 2 - st * w2, w0 + 1 - st * (w2 - 1)
                if (l1 > 10 or l2 > 10):
                    continue
                iloop = 0
                while True and iloop < 100000:
                    x, kx, px = random.randint(2, 5), random.randint(2, 5), random.randint(0, 3)
                    iloop += 1
                    if (l1 <= x + kx - 2 * px and x + kx - 2 * px <= r1):
                        break
                if iloop == 100000:
                    continue
                iloop = 0
                while True and iloop < 100000:
                    y, ky, py = random.randint(2, r2 + 6), random.randint(2, r2 + 6), random.randint(0, 3)
                    iloop += 1
                    if (l2 <= y + ky - 2 * py and y + ky - 2 * py <= r2):
                        break
                if iloop == 100000:
                    continue
                del(m[3 * p])
                del(m[3 * p])
                del(m[3 * p])
                m.insert(3 * p, nn.MaxPool2d((kx, ky), st))
                m.insert(3 * p, nn.ReLU())
                m.insert(3 * p, nn.Conv2d(c0, c2, (x, y), padding = (px, py)))
                st = random.randint(1, min((h1 + 3) // (h0 - 1), (w1 + 3) // (w0 - 1)))
                l1, r1 = h1 + 2 - st * h0, h1 + 1 - st * (h0 - 1)
                l2, r2 = w1 + 2 - st * w0, w1 + 1 - st * (w0 - 1)
                while True:
                    x, kx, px = random.randint(2, r1 + 6), random.randint(2, r1 + 6), random.randint(0, 3)
                    if (l1 <= x + kx - 2 * px and x + kx - 2 * px <= r1):
                        break
                while True:
                    y, ky, py = random.randint(2, r2 + 6), random.randint(2, r2 + 6), random.randint(0, 3)
                    if (l2 <= y + ky - 2 * py and y + ky - 2 * py <= r2):
                        break
                m.insert(3 * p, nn.MaxPool2d((kx, ky), st))
                m.insert(3 * p, nn.ReLU())
                m.insert(3 * p, nn.Conv2d(c1, c0, (x, y), padding = (px, py)))
                flg = False
    return m

def test(model):
    tot = 0
    ac = 0
    model.cuda()
    with torch.no_grad():
        for data, label in te_loader:
            data, label = tensor2cuda(data), tensor2cuda(label)
            output = model(data)
            mylabel = torch.max(output, 1)[1]
            tot += label.size()[0]
            ac += (mylabel == label).sum()
    return 100.0 * ac / tot

def train(structure, epc, lbd_b, lbd_f):
    model = nn.Sequential(*structure)
    opt = torch.optim.Adam(model.parameters(), lr = lbd_b)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epc, lbd_f)
    model.cuda()
    for i in range(epc):
        for data, label in tr_loader:
            data, label = tensor2cuda(data), tensor2cuda(label)
            opt.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, label)
            loss.backward()
            opt.step()
        print(loss)
        sch.step()
    torch.save(model, 'model.pth')
    return test(model)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type = int, default = 512)
    parse.add_argument("--id", type = int, default = 0, help = "gpu id")
    parse.add_argument("--epc", type = int, default = 1000)
    parse.add_argument("--lbd_b", type = float, default = 0.01, help = "lr in the beginning")
    parse.add_argument("--lbd_f", type = float, default = 0.001, help = "lr in the end")
    parse.add_argument("--ckpt", type = str, help = "load saved model for test")
    parse.add_argument("--seed", type = int, default = 2018011309)
    args = parse.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    structure = [Conv2d(3, 6, kernel_size=(2, 2), stride=(1, 1)), ReLU(), MaxPool2d(kernel_size=(2, 1), stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(6, 12, kernel_size=(3, 5), stride=(1, 1), padding=(2, 3)), ReLU(), MaxPool2d(kernel_size=(2, 2), stride=3, padding=0, dilation=1, ceil_mode=False), Conv2d(12, 15, kernel_size=(2, 2), stride=(1, 1)), ReLU(), MaxPool2d(kernel_size=(2, 2), stride=3, padding=0, dilation=1, ceil_mode=False), Reshape(-1, 60), Linear(in_features=60, out_features=120, bias=True), ReLU(), Linear(in_features=120, out_features=84, bias=True), ReLU(), Linear(in_features=84, out_features=10, bias=True)]
    tr_trans = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ])
    te_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ])
    tr_dtset = torchvision.datasets.CIFAR10(root = './', transform = tr_trans)
    te_dtset = torchvision.datasets.CIFAR10(root = './', train = False, transform = tr_trans)
    tr_loader = DataLoader(tr_dtset, batch_size = args.batch_size, shuffle = True, num_workers = 10)
    te_loader = DataLoader(te_dtset, batch_size = args.batch_size, num_workers = 10)
    torch.cuda.set_device(args.id)
    print(train(structure, args.epc, args.lbd_b, args.lbd_f))
