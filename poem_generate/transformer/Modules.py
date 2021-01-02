import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 分母
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q shape: (heads*batch) x len_q x dk
        # k shape: (heads*batch) x len_k x dk
        # v shape: (heads*batch) x len_v x dv
        # mask shape: heads*batch_size, len_q, len_k

        # q 乘 k 转置, attn shape: heads*batch, len_q, len_q

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # 第一维度索引，隔batch_size后数据的mask位置相同， mask为true的位置变成负inf

        attn = self.softmax(attn)
        attn = self.dropout(attn)  # 某些位置随即为０
        # attn shape: batch*heads, len_q, len_q
        # v shape:    heads*batch, len_v, dv
        # 实际上就是 ss * s dv
        output = torch.bmm(attn, v)

        return output, attn
