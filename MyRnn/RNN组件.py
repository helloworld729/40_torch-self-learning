# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/6/6 15:07

import torch
import torch.nn as nn
from torch.nn import utils as nn_utils


class PACKEDRNN(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super(PACKEDRNN, self).__init__()
        # 网络相关的超参数
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def length_check(self, tensor_in, seq_lens):
        (batch_size, L, input_size) = tensor_in.shape  # 2, 4, 3
        assert torch.max(seq_lens) <= L, "输入长度参数(seq_lens)的最大值超过了" \
                                         "模型输入(tensor_in)的第二维度"
        return batch_size, L, input_size

    def forward(self, tensor_in, seq_lens, batchFirst=True):
        # 0、输入相关的超参数
        (batch_size, L, input_size) = self.length_check(tensor_in, seq_lens)

        # 1、倒排索引+恢复索引
        _, dec_idx = torch.sort(seq_lens, dim=0, descending=True)
        _, rec_idx = torch.sort(dec_idx, dim=0, descending=False)

        # 2、获取长度与输入的倒排结果
        orded_seq_lengths = seq_lens.index_select(dim=0, index=dec_idx)
        orded_tensor_in  = tensor_in.index_select(dim=0, index=dec_idx)

        # 3、生成pack_padded对象
        x_packed = nn_utils.rnn.pack_padded_sequence(orded_tensor_in,
                                   orded_seq_lengths, batch_first=batchFirst)

        # 4、RNN初始化：两个维度(输入维度、隐层维度)、一个深度(网络深度)
        rnn = nn.RNN(input_size, self.hidden_size, self.n_layers,
                     batch_first=batchFirst, nonlinearity='relu', bias=False)

        # 5、隐向量初始化：个数约束(网络深度、batchSize)、维度约束(hiddenSize)，
        #                  与是否batchFirst无关
        h0 = torch.randn((self.n_layers*batch_size, self.hidden_size)).view(
            self.n_layers, batch_size, self.hidden_size)

        # 6、返回压缩后的输出上边界和右边界
        y_packed, h_n = rnn(x_packed, h0)

        # 7、pad_packed(输出需要padding，隐层不需要padding)
        y_sorted_padded, length = nn_utils.rnn.pad_packed_sequence(
            y_packed, batch_first=batchFirst)

        # 8、顺序恢复
        right_y = torch.index_select(y_sorted_padded, dim=0, index=rec_idx)
        last_h = torch.index_select(h_n, dim=1, index=rec_idx)

        return right_y, last_h

if __name__ == '__main__':
    # 输入：输入+padding+长度指示
    tensor_in = torch.randn(2, 4, 3)  # b, l, h
    tensor_in[0][2:, ] = torch.zeros((2, 3), dtype=torch.float)  # 输入是float
    seq_lens = torch.tensor([2, 4], dtype=torch.int)             # 长度为int

    # 模型
    module = PACKEDRNN(n_layers=3, hidden_size=5)

    # 输出
    right_y, last_h = module(tensor_in, seq_lens)

    print(right_y.shape)  # torch.Size([2, 4, 5])  # batchSize, seqLen, hiddenSize
    print(last_h.shape)   # torch.Size([3, 2, 5])  # layers, batchSize, hiddenSize

