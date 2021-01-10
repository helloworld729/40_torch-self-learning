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
        # q shape: (heads*batch) x len x dk
        # k shape: (heads*batch) x len x dk
        # v shape: (heads*batch) x len x dv
        # mask shape: heads*batch_size, len, len

        # q 乘 k 转置, attn shape: heads*batch, len, len
        # 那么在每一个batch位，都是一个len*len的方阵，
        # 每一行表示该len位的query结果，即该位置单词和所有单词attention系数
        # 所以，以行为单位，可以理解为 准备好的value的系数
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # mask shape: heads * batch_size, len, len
            # 另外，由于句子的末端，也就是len位比较大的地方，可能是pading的结果
            # 所以，这些位置注意力得分应该为-inf，具体分析如下：
            # 假如句子长度为4，最后一个位置为padding的话就是
            # T T T F
            # T T T F
            # T T T F
            # T T T F
            # 这样以来，说明position4位padidng的位置，接下来mask_fill所有的
            # F为-inf，然后再按照dim=2，进行softmax，那么F的位置能够分到的
            # 权重为0
            # 将该是-inf的位置 填充
            attn = attn.masked_fill(mask, -np.inf)

        # attn：heads*batch_size, len, len 现在dim=2
        attn = self.softmax(attn)
        attn = self.dropout(attn)  # 某些位置随即为０

        # attn shape: batch*heads, len_q, len_q
        # v/output的 shape:    heads*batch, len_v, dv
        # 实际上就是 s, s, dv，每一行的结果都屏蔽了pdding的value位
        # 但是由于本身padding的几行仍然有数据，所以out_put的后几行
        # 可能是无效的，这就需要出去后，硬截断。
        output = torch.bmm(attn, v)
        return output, attn

