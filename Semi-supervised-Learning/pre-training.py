import pdb
import torch
import numpy as np
import time
import random
import myDataset
import Model
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from torch import optim

tqdm.monitor_interval = 0

# Some Global Variables
# Training Parameters
learning_rate = 0.0001
batch_size = 128


Mode = 0  # 0:Encoder-Linear     1:Encoder-Decoder

# Network Parameters
input_size = 1
output_size = 11
num_epoches = 20

display_step = 1

unsuper_trainData = np.load("UNSW-NB15/pretraining.npy",)
unsuper_trainLabel = np.load("UNSW-NB15/pretraining.npy")


train_dataset = myDataset.unsuper_Dataset(unsuper_trainData,
                                          unsuper_trainLabel)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print("Dataset loaded!")

if Mode == 0:
    use_gpu = torch.cuda.is_available()  # GPU enable

    encoder = Model.CNNEncoder(input_size, output_size)

    if use_gpu:
        encoder = encoder.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    loss_function = nn.MSELoss()

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)#, weight_decay = 0.005)
    train_loss = []

    for epoch in range(num_epoches):
        running_loss = 0.0
        avg_loss = 0.0
        for i, data in enumerate(train_loader, 1):
            seq, target = data

            seq = Variable(seq).float()
            target = Variable(target).float()

            if use_gpu:
                seq = seq.cuda()
                target = target.cuda()

            out = encoder(seq)
            loss = loss_function(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * target.size(0)

        if epoch % display_step == 0:
            avg_loss = running_loss / len(train_dataset)
            print('[{}/{}] Loss: {:.6f}'.format(epoch + 1, num_epoches, avg_loss))
            train_loss.append(avg_loss)

            eval_loss = 0.
            eval_acc = 0.

        torch.save(encoder, 'UNSW-NB15/pretrain.pth')


print("Optimization Done!")
t = np.arange(len(train_loss))
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.plot(t, train_loss, color="red", linewidth=2.5,
         linestyle="-", label="Unsupervised Loss")
plt.legend(loc='upper right')
#plt.show()
plt.savefig('Unsupervised.png')

