import pdb
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np


class unsuper_Dataset(torch.utils.data.Dataset):
    def __init__(self, unsuper_trainData, unsuper_trainLabel):
        self.train_data = unsuper_trainData
        self.train_labels = unsuper_trainLabel

    def __getitem__(self, index):
        data, target = self.train_data[index, np.newaxis, :], self.train_labels[index, :]

        return data, target

    def __len__(self):
        return len(self.train_labels)


class super_Dataset(torch.utils.data.Dataset):
    def __init__(self, super_trainData, super_trainLabel, testData, testLabel, train=True):
        self.train_data = super_trainData
        self.train_labels = super_trainLabel
        self.test_data = testData
        self.test_labels = testLabel
        self.train = train

    def __getitem__(self, index):
        if self.train:
            data, target = self.train_data[index, np.newaxis, :], self.train_labels[index]
        else:
            data, target = self.test_data[index, np.newaxis, :], self.test_labels[index]

        return data, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
