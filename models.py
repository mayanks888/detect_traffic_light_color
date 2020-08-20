from __future__ import print_function
import argparse

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import random
from PIL import Image
import torch

import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Color_Net_CNN(nn.Module):
    def __init__(self,output):
        super(Color_Net_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(5 * 5 * 32, 1000)
        # self.fc1 = nn.Linear(800, 1000)
        self.fc2 = nn.Linear(1000, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc1(out))
        out = self.fc3(out)
        # return out
        return F.log_softmax(out, dim=1)


class Color_Net_MLP(nn.Module):
    def __init__(self):
        super(Color_Net_MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)