# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/1/17 16:10

import torch
import torch.nn as nn
import torch.nn.utils as utils
# 学习使用LSTM, 注意没有使用batchFirst
# 情景设定：layers=2，batch为3，嵌入为4，隐层为5，句长为6
layers, batchSize, embeddingSize, hiddenSize = 2, 3, 4, 5

def myRNN(x, originLen, batchFirst=False):
    """
    自定义RNN模型
    :param x:shape seqLen, batchSize, hiddenSize
    :param originLen: 表明batch数据padding之前的原始长度
    :param batchFirst: batch是否作为第一个维度
    :return: y, hn
    """
    # sort
    _, orderLenidx = originLen.sort(dim=0, descending=True)  # 2 0 1
    _, originLenidx = orderLenidx.sort(dim=0)  # 1 2 0
    orderLen = originLen.index_select(0, orderLenidx)  # 6 4 3
    orderx = x.index_select(1, orderLenidx)

    # pack
    xPacked = utils.rnn.pack_padded_sequence(orderx, orderLen, batch_first=batchFirst)
    xPackedData, xPackedBatch = xPacked.data, xPacked.batch_sizes

    # forward
    model = nn.RNN(embeddingSize, hiddenSize, layers, batch_first=batchFirst,
                   nonlinearity="relu", bias=False)
    h0 = torch.randn(layers, batchSize, hiddenSize)
    yPacked, hn = model(xPacked, h0)

    # pad
    yPadded, _ = utils.rnn.pad_packed_sequence(yPacked, batch_first=False)
    y = yPadded.index_select(dim=1, index=originLenidx)
    hn = hn.index_select(dim=1, index=originLenidx)
    return y, hn

def padX(x, batchFirst=False, paddingValue=0):
    """将xpadding成等长的张量"""
    originlen = torch.tensor([t.shape[0] for t in x])
    x = utils.rnn.pad_sequence(x, batch_first=batchFirst,
                               padding_value=paddingValue)
    return x, originlen

# input
x1 = torch.randn(4, embeddingSize)
x2 = torch.randn(3, embeddingSize)
x3 = torch.randn(6, embeddingSize)
x = [x1, x2, x3]

X, originLen = padX(x)
# y是最后一层所有步，hn是最后一步所有层
y, hn = myRNN(X, originLen, batchFirst=False)
print(y)
print(hn)

# 排序原理：假如长度为4 3 6
# 长度倒排索引为 2 0 1-->6 4 3
# 索引正排为     1 2 0-->4 3 6
# 所谓的索引正排就是 为了将倒排的结果恢复，应该怎样选取数据

