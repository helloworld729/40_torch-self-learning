"""
Author:Knoght
Time:2021/2/1
"""
import torch.nn.functional as F
import math, torch, torch.nn as nn
from models.rnn    import LayerNormLSTM
from torch.distributions.normal import Normal
from models.neural import MultiHeadedAttention
from models.neural import PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class EsimMonitor(nn.Module):
    def __init__(self):
        super(EsimMonitor, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(768, eps=1e-6)
        self.m = Normal(torch.FloatTensor([0]), torch.FloatTensor([0.1]))

    def depose2(self, vec1, vec2):
        # batch, seqLen, hiddenSize
        aStandard = vec1 / torch.norm(vec1)  # batch, seqLen, hiddenSize # a方向上的单位向量
        bParaLen = (aStandard * vec2).sum(-1).unsqueeze(-1)
        para = aStandard * bParaLen
        vert = vec2 - para
        para, vert = self.layer_norm(para), self.layer_norm(vert)
        return para, vert

    def forward(self, x, monitor, attention_mask=None):
        q, k, v = monitor, x, x  # batchSize, seqLen, hiddenSize
        mask = attention_mask.eq(0)

        seq_score = F.softmax(  # batchSize, seqLen, seqLen
            (torch.matmul(q, k.permute(0, 2, 1)) / (k.size(2) ** (0.5))).masked_fill(mask.unsqueeze(1), -float('inf')),
            dim=-1)
        # seq_score = seq_score
        seq_attn = torch.matmul(seq_score, v)  # batchSize, seqLen, hiddenSize
        seq_attn = seq_attn.masked_fill(mask.unsqueeze(-1),0)  # padding句子的输出设为0

        q = q.masked_fill(mask.unsqueeze(-1), 0)
        seq_attn_sub_q  = (q - seq_attn)  # batch，hiddenSize
        seq_attn_add_q = (q + seq_attn)  # batch，hiddenSize
        # seq_attn_mean = seq_attn.mean(dim=1).squeeze(1)  # batch，hiddenSize
        # q_mean = q.mean(dim=1).squeeze(1)  # batch，hiddenSize*2
        # 求q方向上的单位向量
        para, vert = self.depose2(q, v)

        out_hidden = torch.cat(
            [seq_attn_sub_q, seq_attn_add_q, para, vert], dim=-1)
        noise = self.m.sample(out_hidden.shape).squeeze(-1)
        if out_hidden.is_cuda:
            noise = noise.cuda()
        out_hidden += noise
        return out_hidden


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo1 = nn.Linear(d_model, 1, bias=True)
        self.wo2 = nn.Linear(d_model * 4, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.monitor = EsimMonitor()

    def forward(self, sensVec, senMask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents, embedSize = sensVec.shape
        pos_emb = self.pos_emb.pe[:, :n_sents]     # 1， n_sents, hiddenSize(768)
        # x = sensVec * senMask[:, :, None].float()  # 多余
        # assert (sensVec == x).data.numpy().all()   # 所有元素相等的时候为True
        x = sensVec + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~senMask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)  # x shape:batchSize, sentCounts, embeddingSize

        # 原始方法
        # sent_scores = self.sigmoid(self.wo1(x))  # sent_scores:batchSize, sentCounts, 1
        # sent_scores = sent_scores.squeeze(-1) * senMask.float()  # sent_scores:batchSize, sentCounts

        # 创新方法1-余弦相似度
        # monitor = torch.sum(x, dim=1) / n_sents  # monitor shape:batchSize, embeddingSize
        # monitor = monitor.unsqueeze(2)  # monitor shape:batchSize, embeddingSize, 1
        # sent_scores = torch.matmul(x, monitor).squeeze(-1)  # sent_scores shape:batchSize, sentCounts,
        # sent_scores = sent_scores.softmax(dim=1)* senMask.float()

        # 创新方法2-avgMonitor
        # monitor = torch.sum(x, dim=1) / n_sents  # monitor shape:batchSize, embeddingSize
        # monitor = monitor.unsqueeze(1).expand(batch_size, n_sents, embedSize)
        # x = torch.cat((x, monitor), dim=2)
        # sent_scores = self.sigmoid(self.wo2(x))  # sent_scores shape:batchSize, sentCounts,
        # sent_scores = sent_scores.squeeze(-1) * senMask.float()

        # 创新方法3-Esim
        monitor = torch.sum(x, dim=1) / n_sents  # monitor shape:batchSize, embeddingSize
        monitor = monitor.unsqueeze(1).expand(batch_size, n_sents, embedSize)
        esim = self.monitor(x, monitor, senMask)  # batchSize, 4*hiddenSize
        sent_scores = self.sigmoid(self.wo2(esim))  # sent_scores shape:batchSize, sentCounts,
        sent_scores = sent_scores.squeeze(-1) * senMask.float()
        return sent_scores



class RNNEncoder(nn.Module):

    def __init__(self, bidirectional, num_layers, input_size,
                 hidden_size, dropout=0.0):
        super(RNNEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn = LayerNormLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self.wo = nn.Linear(num_directions * hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        """See :func:`EncoderBase.forward()`"""
        x = torch.transpose(x, 1, 0)
        memory_bank, _ = self.rnn(x)
        memory_bank = self.dropout(memory_bank) + x
        memory_bank = torch.transpose(memory_bank, 1, 0)

        sent_scores = self.sigmoid(self.wo(memory_bank))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        return sent_scores


"""
想法：在摘要生成的过程中，需要依靠监督向量来表征整个文章的信息
这样的话再给句子打分、选择的时候可以有所参考。
监督向量的生成方式：
1、基于各个时间步的特征向量生成
2、基于自监督隐向量生成

关于句子打分与选择：在打分选择的时候，可以通过计算残余向量来生成剩余句子分数

模型对比：Bert系列

流程梳理：
1、基于bert对文本进行特征向量的抽取
2、基于层级attention抽取每个句子的特征
3、基于每个token/句子的特征向量构建文档级别监督向量

4、打分选择的时候有两种思路
4.1：直接基于监督向量和句子向量建模打分(例如MLP、内积)
4.2：将打分和选择融合在一起：需要借助RNN动态的进行选择
     并且可以考虑使用监督向量/残余向量(监督向量与介解码语境做差)

创新点：监督向量、参与向量
注意：去重冗余(block_trick、覆盖机制)
################################################################
baseLine：基于句子向量构建监督向量，然后分立解码
"""