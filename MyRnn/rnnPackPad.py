# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/1/17 16:10

import torch
import torch.nn as nn
import torch.nn.utils as utils
# 学习使用LSTM, 注意没有使用batchFirst
# 情景设定：layers=2，batch为3，嵌入为4，隐层为5，句长为6
layers, batchSize, embeddingSize, hiddenSize, seqLen = 2, 3, 4, 5, 6

# input
x = torch.randn(batchSize, seqLen, embeddingSize)
x[0][-2:] = torch.zeros(2, 4)
x[1][-3:] = torch.zeros(3, 4)
x = x.transpose(0, 1).contiguous()
assert x.shape == (seqLen, batchSize, embeddingSize)
originLen = torch.tensor([4, 3, 6])

# sort
_, orderLenidx = originLen.sort(dim=0, descending=True)  # 2 0 1
_, originLenidx = orderLenidx.sort(dim=0)  # 1 2 0
orderLen = originLen.index_select(0, orderLenidx)  # 6 4 3
orderx = x.index_select(1, orderLenidx)

# pack
xPacked = utils.rnn.pack_padded_sequence(orderx, orderLen)
xPackedData, xPackedBatch = xPacked.data, xPacked.batch_sizes

# forward
model = nn.RNN(embeddingSize, hiddenSize, layers, batch_first=False,
               nonlinearity="relu", bias=False)
h0 = torch.randn(layers, batchSize, hiddenSize)
yPacked, hn = model(xPacked, h0)

# pad
yPadded, _ = utils.rnn.pad_packed_sequence(yPacked, batch_first=False)
y = yPadded.index_select(dim=1, index=originLenidx)
hn = hn.index_select(dim=1, index=originLenidx)
assert y.shape == (seqLen, batchSize, hiddenSize)
assert hn.shape == (layers, batchSize, hiddenSize)


# 排序原理：假如长度为4 3 6
# 长度倒排索引为 2 0 1-->6 4 3
# 索引正排为     1 2 0-->4 3 6
# 所谓的索引正排就是 为了将倒排的结果恢复，应该怎样选取数据

