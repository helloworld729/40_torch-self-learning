''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)  # 数据量没有变化，看做降维加head的过程
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)  # 类的实例化
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        """batch_size, seq_len, embedding_size"""

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()  # batch_size, len_q(seq_len), d_model
        sz_b, len_k, _ = k.size()  # batch_size, len_q(seq_len), d_model
        sz_b, len_v, _ = v.size()  # batch_size, len_q(seq_len), d_model

        residual = q  # q作为残差项

        # 多头线性转换
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # batch_size, len_q(seq_len), heads, d_k
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # batch_size, len_k(seq_len), heads, d_k
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # batch_size, len_v(seq_len), heads, d_v

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (heads*batch) x len_q x dk  # 分头降维 由d_model-->dk  由len，d_model-->len，d_k，padding的位置仍然为0
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (heads*batch) x len_k x dk  # 分头降维 由d_model-->dk  由len，d_model-->len，d_k，padding的位置仍然为0
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (heads*batch) x len_v x dv  # 分头降维 由d_model-->dk  由len，d_model-->len，d_k，padding的位置仍然为0

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x .. 对于每一个头而言，数据没变所以掩码可以直接复制 tile函数
        output, attn = self.attention(q, k, v, mask=mask)  # 调用前向函数 output是最终的输出， attn是乘v之前

        output = output.view(n_head, sz_b, len_q, d_v)  # heads*batch, len_q, d_v  --->  heads， batch, len_q, d_v
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # batch_size， seq_len， (heads*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # 输入通道：d_modle， 输出通道：d_inner（前向隐藏），卷积核为1
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # 输入通道：d_inner（前向隐藏）， 输出通道：d_inner，卷积核为1
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
