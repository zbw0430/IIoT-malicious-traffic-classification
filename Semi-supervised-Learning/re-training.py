import pdb
import torch
import numpy as np
import myDataset
import Model
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim import lr_scheduler
import tqdm
import torch.nn as nn
from torch import optim

# Some Global Variables
# Training Parameters
learning_rate = 0.0001
epsilon = 1e-08
weight_decay = 1e-8
batch_size = 128

# Network Parameters
input_size = 1
output_size = 11
num_classes = 2     #for supervised learning
num_epoches = 20

display_step = 1

use_gpu = torch.cuda.is_available()

tqdm.monitor_interval = 0

total_acc_single = 0
total_acc = np.zeros(num_classes)
total_pre = np.zeros(num_classes)
total_rec = np.zeros(num_classes)
total_f1 = np.zeros(num_classes)
full_acc = []
full_pre = []
full_rec = []
full_f1 = []

(super_trainData, super_trainLabel, testData, testLabel) = np.load("UNSW-NB15/retraining.npy", allow_pickle=True)

train_dataset = myDataset.super_Dataset(super_trainData,
                                        super_trainLabel, testData, testLabel, train=True)

test_dataset = myDataset.super_Dataset(super_trainData,
                                       super_trainLabel, testData, testLabel, train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
print("Dataset loaded!")


encoder = Model.CNNEncoder(input_size, output_size)

# Load the pre-trained model here:
pretrained_encoder = torch.load('UNSW-NB15/pretrain.pth')
encoder.load_state_dict(pretrained_encoder.state_dict())

# For transfer learning, the convolutional layers can be fixed
# layers freeze
fixed = True
if fixed:
    for param in encoder.cnnseq.parameters():
        param.requires_grad = False

encoder.reggresor = nn.Sequential()

decoder = nn.Sequential(
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(inplace=True),
    nn.Linear(32, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(inplace=True),
    nn.Linear(16, 8),
    nn.BatchNorm1d(8),
    nn.ReLU(inplace=True),
    nn.Linear(8, num_classes))

transferedModel = nn.Sequential(encoder, decoder)

if use_gpu:
    transferedModel = transferedModel.cuda()

# loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()

# When freezing layer, this only works
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,transferedModel.parameters()), lr=learning_rate,eps=epsilon, weight_decay=weight_decay)

train_acc = []
test_acc = []
best_test_acc = 0.

for epoch in range(num_epoches):
    running_loss = 0.0
    running_acc = 0.0
    avg_loss = 0.0
    avg_acc = 0.0
    total_target = 0

    for i, data in enumerate(train_loader, 1):
        seq, target = data
        seq = Variable(seq).float()
        target = Variable(target).long()

        if use_gpu:
            seq = seq.cuda()
            target = target.cuda()

        out = transferedModel(seq)
        loss = loss_function(out, target)

        # prediction
        _, pred = torch.max(out, 1)
        total_target += len(pred)
        num_correct = (pred == target).sum()

        running_acc += num_correct.item()
        running_loss += loss.item() * target.size(0)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % display_step == 0 or epoch==num_epoches-1:
        avg_loss = running_loss / (total_target)
        avg_acc = running_acc / (total_target)

        print('Loss: {:.6f}, Acc: {:.6f}'.format(
                    avg_loss,
                    avg_acc))

        train_acc.append(avg_acc)

        # test
        transferedModel.eval()
        eval_loss = 0.
        eval_acc = 0.
        for i, data in enumerate(test_loader, 1):
            seq, target = data
            seq = Variable(seq).float()
            target = Variable(target).long()

            if use_gpu:
                seq = seq.cuda()
                target = target.cuda()

            out = transferedModel(seq)
            loss = loss_function(out, target)

            # prediction
            eval_loss += loss.item() * target.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == target).sum()
            eval_acc += num_correct.item()

        avg_acc = eval_acc / (len(test_dataset))
        test_acc.append(avg_acc)
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_dataset)), avg_acc))

        if avg_acc > best_test_acc:
            best_test_acc = avg_acc
            torch.save(transferedModel, 'UNSW-NB15/best.pth')

print("Optimization Done!")
t = np.arange(len(train_acc))

np.set_printoptions(precision=3)
TP = np.zeros(num_classes)
TN = np.zeros(num_classes)
FP = np.zeros(num_classes)
FN = np.zeros(num_classes)

transferedModel = torch.load('UNSW-NB15/best.pth')
transferedModel.eval()
eval_loss = 0
eval_acc = 0
for i, data in enumerate(test_loader, 1):
    seq, target = data
    seq = Variable(seq).float()
    target = Variable(target).long()

    if use_gpu:
        seq = seq.cuda()
        target = target.cuda()

    out = transferedModel(seq)
    loss = loss_function(out, target)

    # prediction
    eval_loss += loss.item() * target.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == target).sum()
    eval_acc += num_correct.item()

    for i in range(len(pred)):
        if target[i] == pred[i]:
            TP[target[i]] += 1
            for j in range(num_classes):
                if j != target[i]:
                    TN[j] += 1
        elif target[i] != pred[i]:
            FP[pred[i]] += 1
            FN[target[i]] += 1
avg_acc = eval_acc / (len(test_dataset))

Accuracy = np.true_divide(TP+TN, TP+TN+FP+FN)
precision = np.true_divide(TP, TP+FP)
recall = np.true_divide(TP, TP+FN)
F1 = 2 * np.true_divide(np.multiply(precision,recall), precision+recall)
print(TP,TN)
print(FP,FN)
print("Accuracy: ", Accuracy)
print("precision: ", precision)
print("recall: ", recall)
print("F1: ", F1)
full_acc.extend(Accuracy)
full_pre.extend(precision)
full_rec.extend(recall)
full_f1.extend(F1)
total_acc += Accuracy
total_pre += precision
total_rec += recall
total_f1 += F1
total_acc_single += avg_acc

print("Evaluation is done!")
print("Accuracy: ", full_acc)
print("precision: ", full_pre)
print("recall: ", full_rec)
print("F1: ", full_f1)
