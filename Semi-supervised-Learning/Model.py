import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch.nn import init


class CNNEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.output_size = output_size

        self.cnnseq = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )
        self.reggresor = nn.Sequential(
            nn.Linear(64, self.output_size)
        )

    def forward(self, images):
        code = self.cnnseq(images)
        code = code.view([images.size(0), -1])
        if self.reggresor:
            code = self.reggresor(code)
            code = code.view([code.size(0), self.output_size])
        return code


class Decoder(nn.Module):

    def __init__(self, in_dim, output_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.linear3 = nn.Linear(in_dim, output_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(x))
        out = self.linear3(out)
        out = F.softmax(out, dim=1)
        return out


