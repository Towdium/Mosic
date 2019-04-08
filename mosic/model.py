from torch import nn
import torch
import numpy as np
import math

'''
This file contains all the models used in experiments.
The models can be dynamically sized
'''

class Alex(nn.Module):
    def __init__(self, shape, outputs, size=512):
        super(Alex, self).__init__()
        self.size = size
        tmp = []
        conv = []
        height = int(math.log2(min(shape[1:]))) - 1
        diff = int(size // height)
        for i in range(height - 1):
            tmp.append(nn.Conv2d(size - (i + 1) * diff, size - i * diff, 3))
        tmp.append(nn.Conv2d(shape[0], size - (height - 1)*diff, 3))

        for i in reversed(tmp):
            conv.append(i)
            conv.append(nn.ReLU())
            conv.append(nn.MaxPool2d(2))
        tmp = nn.Sequential(*conv)
        x = torch.Tensor(np.zeros((1, *shape)))
        x = tmp.forward(x)

        coef = (size / outputs) ** (1 / 3)
        fc = [
            nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], int(size / coef)),
            nn.ReLU(),
            nn.Linear(int(size / coef), int(outputs * coef)),
            nn.ReLU(),
            nn.Linear(int(outputs * coef), outputs)
        ]

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    @staticmethod
    def identifier():
        return 'alex'


class Inception(nn.Module):
    # modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
    class Node(nn.Module):
        def __init__(self, input, output):
            super(Inception.Node, self).__init__()
            # 1x1 conv branch
            self.b1 = nn.Sequential(
                nn.Conv2d(input, output // 4, kernel_size=1),
                nn.BatchNorm2d(output // 4),
                nn.ReLU(True),
            )

            # 1x1 conv -> 3x3 conv branch
            self.b2 = nn.Sequential(
                nn.Conv2d(input, output // 4, kernel_size=1),
                nn.BatchNorm2d(output // 4),
                nn.ReLU(True),
                nn.Conv2d(output // 4, output // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(output // 2),
                nn.ReLU(True),
            )

            # 1x1 conv -> 5x5 conv branch
            self.b3 = nn.Sequential(
                nn.Conv2d(input, output // 16, kernel_size=1),
                nn.BatchNorm2d(output // 16),
                nn.ReLU(True),
                nn.Conv2d(output // 16, output // 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(output // 8),
                nn.ReLU(True),
            )

            # 3x3 pool -> 1x1 conv branch
            self.b4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(input, output // 8, kernel_size=1),
                nn.BatchNorm2d(output // 8),
                nn.ReLU(True),
            )

        def forward(self, x):
            y1 = self.b1(x)
            y2 = self.b2(x)
            y3 = self.b3(x)
            y4 = self.b4(x)
            return torch.cat((y1, y2, y3, y4), 1)

    def __init__(self, shape, outputs, size=512):
        super(Inception, self).__init__()
        height = int(math.log2(min(shape[1:]))) - 1
        diff = (size // 16 - 2) // height // 2 * 16
        self.pre = nn.Sequential(
            nn.Conv2d(shape[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        tmp = []
        dim = size - height * 2 * diff
        maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        for i in range(height):
            tmp.append(Inception.Node(32 if i == 0 else dim, dim + diff))
            tmp.append(Inception.Node(dim + diff, dim + 2 * diff))
            tmp.append(maxpool)
            dim += 2 * diff
        tmp = tmp[:-1]
        x = torch.Tensor(np.zeros((1, *shape)))
        x = self.pre(x)
        x = nn.Sequential(*tmp)(x)
        tmp.append(nn.AvgPool2d((x.shape[2], x.shape[3])))
        self.layers = nn.Sequential(*tmp)
        self.linear = nn.Linear(size, outputs)

    def forward(self, x):
        x = self.pre(x)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    @staticmethod
    def identifier():
        return 'inception'


class Directional(nn.Module):
    def __init__(self, shape, outputs, size=512):
        super(Directional, self).__init__()
        self.size = size
        steps = (int(math.log(shape[1], 2)) - 2, int(math.log(shape[2] / 32, 2)))
        diff = size // (steps[0] + steps[1])
        tmp = []
        for i in range(steps[1]):
            tmp += [
                nn.MaxPool2d((1, 2)),
                nn.ReLU(),
                nn.Conv2d(size - diff, size, (1, 3))
            ]
            size -= diff
        for i in range(steps[0] - 1):
            tmp += [
                nn.MaxPool2d((2, 1)),
                nn.ReLU(),
                nn.Conv2d(size - diff, size, (3, 1))
            ]
            size -= diff
        tmp += [
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(shape[0], size, (3, 1))
        ]
        tmp = tmp[::-1]

        size = self.size
        for i in range(int(math.log2(size / 4))):
            tmp += [
                nn.Conv2d(size, size // 2, 1),
                nn.ReLU(),
            ]
            size //= 2

        self.conv = nn.Sequential(*tmp)
        x = torch.Tensor(np.zeros((1, *shape)))
        x = self.conv.forward(x)

        size = 4 * x.shape[2] * x.shape[3]
        inter = int((size * outputs) ** 0.5)
        tmp = [
            nn.Linear(size, inter),
            nn.ReLU(),
            nn.Linear(inter, outputs)
        ]
        self.fc = nn.Sequential(*tmp)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    @staticmethod
    def identifier():
        return 'directional'
