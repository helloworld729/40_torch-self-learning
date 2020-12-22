# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2020/12/22 20:56

import torch
import torch.nn as nn
import torch.nn.functional as F
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # hidden size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # seq_len，batch_size，input_size
        # layers,  batch_size, hidden_size, hidden由下面的函数确定，
        embedded = self.embedding(input).view(1, 1, -1)  # input作为index，从embedding中取出向量 1 16
        output = embedded  # shape：1 1 16
        # out_put:seq_len, batch_size, hidden_size
        # hidden :layers,  batch,      hidden_size
        output, hidden = self.gru(output, hidden)  # 两个输出的shape都是 1 1 16
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 字典的长度和嵌入的维度
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)  # 1 1 16

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # attn_weights是 1*10的Tensor，10对应是设置的句子长度上限
        # 这个上限一方面是削减数据集，另一方面是防止输出不能停止
        # attn_weights是1*10->1*1*10，encoder_outputs是10*16->1*10*16
        # bmm是矩阵乘法，输出 1*1*16，在这里本质上是对编码输出的加权求和
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)  # shape:1*32
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)  # shape：1*1*16
        output, hidden = self.gru(output, hidden)  # 两个Tensor的shape都是1*1*16

        output = F.log_softmax(self.out(output[0]), dim=1)  # shape:1*27
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

