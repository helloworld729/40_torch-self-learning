# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2020/12/23 20:20
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants

def get_non_pad_mask(seq):
    # True:有字的位置， False：padding的位置
    assert seq.dim() == 2
    return seq.eq(Constants.PAD)

seq = torch.randn(2, 5)
seq[0][3:] = torch.tensor([0, 0])
print(seq)
print(get_non_pad_mask(seq))
print("----------------------------------------------------------")
# tensor([[ 0.3639,  1.8589, -0.1482,  0.0000,  0.0000],
#         [-1.6303,  0.6182, -1.1017,  1.1021, -0.7764]])
# tensor([[False, False, False,  True,  True],
#         [False, False, False, False, False]])


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)  # 返回True/False [batch_size, len_k]
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # batch_size, len_q, len_k
    return padding_mask

seq_k = torch.randn(2, 5)
seq_q = torch.randn(2, 5)
seq_k[0][3:] = torch.tensor([0, 0])

print(get_attn_key_pad_mask(seq_k, seq_q))
# 分析：k是查询向量，一个有效长度为3，另一个满编
# q 和 k 相乘 求得系数，用mask去掉无效的位置
# tensor([[[False, False, False,  True,  True],
#          [False, False, False,  True,  True],
#          [False, False, False,  True,  True],
#          [False, False, False,  True,  True],
#          [False, False, False,  True,  True]],
#
#         [[False, False, False, False, False],
#          [False, False, False, False, False],
#          [False, False, False, False, False],
#          [False, False, False, False, False],
#          [False, False, False, False, False]]])

print("----------------------------------------------------------")
mask = torch.randn(2, 3, 3)
print(mask)
print(mask.shape)
mask = mask.repeat(2, 1, 1)
print(mask)
print(mask.shape)
# tensor([[[ 0.6985, -0.1081, -0.6627],
#          [ 0.2367,  0.3207,  0.5775],
#          [-0.1353,  0.0506,  0.1565]],
#
#         [[ 0.0876, -0.4406, -0.4765],
#          [-0.6393, -0.4506, -1.0846],
#          [-1.1942,  0.1008,  0.6887]]])
# torch.Size([2, 3, 3])

# tensor([[[ 0.6985, -0.1081, -0.6627],
#          [ 0.2367,  0.3207,  0.5775],
#          [-0.1353,  0.0506,  0.1565]],
#
#         [[ 0.0876, -0.4406, -0.4765],
#          [-0.6393, -0.4506, -1.0846],
#          [-1.1942,  0.1008,  0.6887]],
#
#         [[ 0.6985, -0.1081, -0.6627],
#          [ 0.2367,  0.3207,  0.5775],
#          [-0.1353,  0.0506,  0.1565]],
#
#         [[ 0.0876, -0.4406, -0.4765],
#          [-0.6393, -0.4506, -1.0846],
#          [-1.1942,  0.1008,  0.6887]]])
# torch.Size([4, 3, 3])

a = torch.randn((2, 2, 3, 4))
b = torch.randn((2, 2, 4, 5))
# c = torch.matmul(a, b)  # 任意维度矩阵相乘
# d = torch.bmm(a, b)     # 只针对3维以下矩阵
# e = torch.mm(a, b)      # 只针对2维以下矩阵
# 另外 mul、multiply、*都是element-wise相乘，可以广播

############################## repeat 函数 ############################
import numpy as np
a = np.linspace(1, 12, 12)
a = torch.from_numpy(a).view(2, 2, 3)
print("############################## repeat 函数 ############################")
# print(a)
# print(a.repeat(2, 1, 1))  #
# tensor([[[ 1.,  2.,  3.],
#          [ 4.,  5.,  6.]],
#
#         [[ 7.,  8.,  9.],
#          [10., 11., 12.]]], dtype=torch.float64)

# tensor([[[ 1.,  2.,  3.],
#          [ 4.,  5.,  6.]],
#
#         [[ 7.,  8.,  9.],
#          [10., 11., 12.]],
#
#         [[ 1.,  2.,  3.],
#          [ 4.,  5.,  6.]],
#
#         [[ 7.,  8.,  9.],
#          [10., 11., 12.]]], dtype=torch.float64)
# 在原矩阵之后追加的方式repeat，而不是内部嵌入

print("# ################################# view 降维测试 ###################################")
# a = torch.randn(2,2,3)
# print(a)
# b = a.view(4, 3)
# print(b)
# tensor([[[ 0.1862,  1.4143,  0.9531],
#          [ 1.1060, -0.4529, -0.2657]],
#
#         [[-0.4519,  2.2953,  1.0560],
#          [ 0.4196, -0.6357,  1.5464]]])
# tensor([[ 0.1862,  1.4143,  0.9531],
#         [ 1.1060, -0.4529, -0.2657],
#         [-0.4519,  2.2953,  1.0560],
#         [ 0.4196, -0.6357,  1.5464]])

print("# ################################# permute 测试 ###################################")
# a = np.linspace(1, 20, 20)
# a = torch.from_numpy(a).view(2, 2, 5)  # heads, batch, len
# b  = a.permute(2, 0, 1)  # len, heads, batch
# print(a)
# print(b)

print("# ################################# 三角矩阵 测试 ###################################")
# def get_subsequent_mask(seq):
#     ''' For masking out the subsequent info. '''
#
#     sz_b, len_s = seq.size()
#     subsequent_mask = torch.triu(
#         torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
#     subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
#
#     return subsequent_mask
#
# seq = torch.randn(2, 12)
# print(get_subsequent_mask(seq))
#
# tensor([[[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
#
#         [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
# 第一个step只能看到一个单词，第二个step只能看到两个单词
print("# ######################################### 中文分词 # #########################################")
# a = "你好，这里是中国"
# a = [i for i in a]
# print(a)

