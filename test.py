import torch
from torch.nn import *
import torch.nn as nn
import random

a = [1, 2, 3, 4, 5]
del(a[1])

class Reshape(Module):
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

x = [Conv2d(3, 16, kernel_size=(2, 2), stride=(1, 1)), 
ReLU(), 
MaxPool2d(kernel_size=(2, 1), stride=2, padding=0, dilation=1, ceil_mode=False), 
Conv2d(16, 22, kernel_size=(3, 2), stride=(1, 1), padding=(3, 2)), 
ReLU(), 
MaxPool2d(kernel_size=(2, 3), stride=1, padding=0, dilation=1, ceil_mode=False), 
Conv2d(22, 16, kernel_size=(4, 3), stride=(1, 1), padding=(2, 2)), 
ReLU(), 
MaxPool2d(kernel_size=(2, 3), stride=3, padding=0, dilation=1, ceil_mode=False), 
Conv2d(16, 32, kernel_size=(4, 4), stride=(1, 1)), 
ReLU(), 
MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False), 
Reshape(), 
Linear(in_features=1152, out_features=120, bias=True), 
ReLU(), 
Linear(in_features=120, out_features=84, bias=True), 
ReLU(), 
Linear(in_features=84, out_features=10, bias=True)]

print(calcshape(x))
