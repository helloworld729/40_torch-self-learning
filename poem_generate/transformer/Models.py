''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

# mask一是在attention权重的时候起作用，另外实在attention计算完之后
# 把padding的位置截断起作用。不过，在每一个子层有必要做none_pad_mask
# 截断吗
def get_non_pad_mask(seq):
    # 有字的位置:1， padding的位置:0
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' 位置编码 Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)  # 返回True/False [batch_size, len_k]
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # batch_size, len_q, len_k
    return padding_mask

# 上三角掩码
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' 多编码层stack A encoder model with self attention mechanism. '''
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        # no_pad_mask -> padding的->0， attn_mask -> v的系数为0(softmax -inf)

        # batch_size, len_q, len_k  (padding位置为True)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)

        # batch_size, seq_len, 1    (padding位置为0)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # batch_size，seq_len，embedding_size
        # embedding 后的开头和结束都相同（BOS, EOS），但是加上位置编码后末尾的vector不同
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            # attention系数enc_slf_attn，可有可无
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # non_pad_mask：batch_size, seq_len, 1
        non_pad_mask = get_non_pad_mask(tgt_seq)
        # sub-seq shape：batch_size, seq_len, seq_len
        # padding 的位置为1
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)

        # key_mask shape：batch_size, len, len均是解码长度 padding的位置为1
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)

        # self attention mask：batch_size，len_q，len_q，是一个下三角矩阵，而且如果有padding
        # 的话，slf_attn_mask_keypad右侧也会显示为True->1，两者相加后可能是一个下侧为矩形
        # 上侧为三角形的形状，也就是说下三角被提前阶段了，下侧的数据实际上会通过none_pad_mask
        # 硬编码处理为0，也就是说三角形的高度实际上就是训练集解码句子实际的长度。
        slf_attn_mask = (slf_attn_mask_keypad.type_as(slf_attn_mask_subseq) + slf_attn_mask_subseq).gt(0)

        # 现在的query是解码侧，key和value是编码侧，所以attn的shape为
        # batch，len_输出，len_输入,每一行仍然是对不同编码步的权重
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=False,
            emb_src_tgt_weight_sharing=False):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            # 两者的shape都是：d_model，n_tgt_vocab (输出侧的单词数量)
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        # 关于这一块，为什么两侧的语言embedding要共享权重呢？因为如果是同一种语言的
        # 例如文本摘要任务，自然需要共享，假如是英文和德文的翻译任务，由于两者都是
        # 日耳曼语系，所以字词嵌入也可以共享，但是假如是中英翻译的话，大可不必。
        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            # weight:就是词向量矩阵
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]  # 删除最后一列

        enc_output, *_ = self.encoder(src_seq, src_pos)
        # 因为只返回了一个数据，所以使用单个返回
        # dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        # dec_output shape：batch，len，d_model
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        # seq_logit shape：shape：batch，len，n_target_vocab
        # 疑惑：为甚么这里要缩放
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


