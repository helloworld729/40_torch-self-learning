import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import copy
import math

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 返回attention的结果与权重分布
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # 返回多头attention后的结果
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 第一子层：归一化+多头attention+残差
        # lambda整体作为SublayerConnection forward的sublayer
        # 先归一化，再进网络，出来求残差
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二子层：归一化+前向+残差
        # self.feed_forward作为SublayerConnection forward的sublayer
        # 先归一化，再进网络，出来求残差
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SimpleEncoder(nn.Module):
    """
    takes (batch_size, seq_len, embed_dim) as inputs
    calculate MASK, POSITION_ENCODING
    """
    def __init__(self, embed_dim, head=4, layer=1, dropout=0.1):
        super(SimpleEncoder, self).__init__()
        d_ff = 2 * embed_dim

        self.position = PositionalEncoding(embed_dim, dropout)
        attn = MultiHeadedAttention(head, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff)
        self.encoder = Encoder(EncoderLayer(embed_dim, attn, ff, dropout), layer)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-2)
        x = self.position(x)
        x = self.encoder(x, mask)
        return x

# if __name__ == '__main__':
#     encoder = SimpleEncoder(300, 4, 2)
#     # batch, docLen, hiddenSize
#     inputs = torch.zeros(1000, 50, 300)
#     mask = torch.ones(1000, 50)
#     mask[:10, 30:] = 0
#     mask[20:30, 20:] = 0
#     print(mask)
#     lens  = [10] * 1000
#     out = encoder(inputs, mask)
#     print(out.size())
#     print(out[0])
#     print(out[-1])

# tensor([[1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         ...,
#         [1., 1., 1.,  ..., 1., 1., 1.],
#         [1., 1., 1.,  ..., 1., 1., 1.],
#         [1., 1., 1.,  ..., 1., 1., 1.]])
# torch.Size([1000, 50, 300])
# tensor([[ 0.9087, -0.2573,  1.8975,  ...,  0.7703, -0.1817,  1.4127],
#         [ 1.2165, -1.1043,  1.6587,  ...,  0.8845, -0.5554,  1.5407],
#         [ 0.1893, -2.8588,  1.8796,  ...,  0.6898, -0.0979,  1.0685],
#         ...,
#         [ 0.4347, -2.4928,  1.3567,  ...,  1.3670, -0.5064,  1.2395],
#         [-0.6836, -1.5751,  2.2632,  ...,  0.9626, -0.3131,  1.5531],
#         [-0.8954, -1.2491,  1.9622,  ...,  1.4277, -0.4073,  1.2929]],
#        grad_fn=<SelectBackward>)
# tensor([[ 0.3866, -0.3584,  0.6327,  ...,  1.3499, -0.7031,  1.6593],
#         [ 1.0766, -0.8731,  2.0193,  ...,  1.4657, -0.1532,  2.0208],
#         [ 1.9742, -1.8697,  1.8364,  ...,  1.2257, -0.4974,  1.6724],
#         ...,
#         [ 0.1173, -2.4423,  0.7331,  ...,  1.4184, -0.0257,  1.6040],
#         [-0.3734, -2.4962,  0.7725,  ...,  1.1138, -0.3450,  1.3697],
#         [-1.1002, -1.2486,  1.6939,  ...,  0.9329, -0.6567,  0.5855]],
#        grad_fn=<SelectBackward>)