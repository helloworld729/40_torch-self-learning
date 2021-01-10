''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"

# 单层编码函数
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)  # 多头attention + 残差 + norm
        # mask的shape： batch_size, seq_len, 1
        # enc_output的shape：batch_size，seq_len，d_model
        # 所以mask会在最后一维广播，padding的位置整个d_model都广播为0
        enc_output *= non_pad_mask

        # 前向 + 残差 + norm
        # enc_output的shape：batch_size，seq_len，d_model
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn     = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn      = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # 首先进行下侧的attention计算，特点是既有pading的截断，又有信息的掩盖，两者结合构成attention mask
        # 实际上这一块的attention是解码句子的self-attention也就是说，在不同解码时间步，做了attention
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        # 数据读入 以下两个attention虽然使用的mask不同，
        # 但是都是多头attention，mask都是在q乘以k之后使用，
        # 把输出作为查询向量k, 输入作为key和value向量
        # dec_output的shape是：batch，seq_len，d_model，对于decode而言，已经是内部各个时间位的attention
        # 而且做了下部的截断，然后做交互的attention，解码侧作为query
        dec_output, dec_enc_attn = self.enc_dec_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)  # 前向
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

