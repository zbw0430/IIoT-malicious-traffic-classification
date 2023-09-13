import pdb
import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 设置随机数种子
np.random.seed(20)


# 加载csv文件，返回data和label
def loadData():
    # Feature columns and label column
    features = ['dur','sbytes','dbytes','sttl','dttl','sloss','dloss','sload','dload','spkts','dpkts']
    label = ['label']

    # Read csv file
    train_df = pd.read_csv('UNSW-NB15/UNSW_NB15_training-set.csv', usecols=features+label)
    test_df = pd.read_csv('UNSW-NB15/UNSW_NB15_testing-set.csv', usecols=features+label)

    # Split feature and label
    train_data = train_df.values[:,:-1]
    train_label = train_df.values[:,-1]
    test_data = test_df.values[:,:-1]
    test_label = test_df.values[:,-1]

    # 输入的特征维度
    Channel_size = len(features)

    return train_data, train_label, test_data, test_label


def convertDataUnsupervised(data):
    # shuffle the data
    num_samples = data.shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)

    data = data[perm]

    unsupervised_train = np.arange(num_samples)
    unsupervised_train = perm[unsupervised_train]

    unsuper_trainData = data[unsupervised_train, :]

    return unsuper_trainData


def convertDataSupervised(superdata, superlabels, supertestdata, supertestlabels):
    # shuffle the data
    num_samples2 = superlabels.shape[0]
    num_samples3 = supertestlabels.shape[0]
    assert num_samples2 == superdata.shape[0]
    perm2 = np.arange(num_samples2)
    perm3 = np.arange(num_samples3)
    np.random.shuffle(perm2)
    np.random.shuffle(perm3)

    superlabels = superlabels[perm2]
    superdata = superdata[perm2]
    supertestlabels = supertestlabels[perm3]
    supertestdata = supertestdata[perm3]

    return (superdata, superlabels, supertestdata, supertestlabels)


if __name__ == "__main__":
    (data, label, data1, label1) = loadData()

    ss = StandardScaler()
    ss.fit(data)
    data = ss.transform(data)
    data1 = ss.transform(data1)

    pretrain_data = np.concatenate((data, data1), axis=0)
    pretrain_data = convertDataUnsupervised(pretrain_data)

    (train_data, train_label, test_data, test_label) = convertDataSupervised(data, label, data1, label1)

    np.save("UNSW-NB15/pretraining.npy", pretrain_data)
    np.save("UNSW-NB15/retraining.npy", (train_data, train_label, test_data, test_label))
