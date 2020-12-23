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
        # (heads*batch) x len_q x dk  # 分头降维 由d_model-->dk  由len，d_model-->len，d_k，padding的位置仍然为0

        attn = torch.bmm(q, k.transpose(1, 2))  # q 乘 k 转置  heads*batch, len_q, len_q
        attn = attn / self.temperature          # 除以分母

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # 第一维度索引，隔batch_size后数据的mask位置相同， mask为true的位置变成负inf

        attn = self.softmax(attn)
        attn = self.dropout(attn)  # 某些位置随即为０
        output = torch.bmm(attn, v)  # 下部余孽  heads*batch, len_q, len_q * heads*batch, len_q, d_v

        return output, attn
