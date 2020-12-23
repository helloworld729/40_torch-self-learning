import numpy as np
from RNN_utils import *
from RNN_ import *
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

batch_size = 4
lr_init = 0.01
MAX_EPOCH = 10

# ----------------------------------------------数据-------------------------------------------------------------------
train_dataset = MyDataset()
test_dataset = MyDataset()

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # len(train_loader)=100
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size)
# ----------------------------------------------模型-------------------------------------------------------------------
this_net = LSTM_single_direction(num_layers=1)
this_net.initialize_weights()
this_net.zero_grad()
# --------------------------------------------损失、优化---------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(this_net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# ----------------------------------------------训练-------------------------------------------------------------------
if __name__ == '__main__':
    for epoch in range(MAX_EPOCH):
        correct = 0.0
        total = 0.0
        for index, data in enumerate(train_loader):
            data_x, label = data  # [batch, seq, bedding]  (batch,)
            optimizer.zero_grad()
            y_hat = this_net(input_data=data_x, hid=None)  # 此处没有启用初始化,应该用元组初始化（h0，c0）
            loss = criterion(y_hat, label)  # [batch,class]float32, (batch,)int64
            loss.backward()
            optimizer.step()

            _, predict = torch.max(y_hat.squeeze(), dim=1)
            total += label.size(0)
            correct += (predict == label).squeeze().sum()
        if epoch % 2 == 0:
            print('epoch:{:<2}/{:<3} Acc{:<5.2f}'.format(epoch, MAX_EPOCH, correct/total))




