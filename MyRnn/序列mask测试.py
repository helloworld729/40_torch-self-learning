import torch
import numpy as np
import torch.nn as nn
from torch.nn import utils as nn_utils

batch_size  = 2
hidden_size = 5
# embedding dimension
input_size = 3
n_layers   = 2

print("1.-------------------------- 数据 --------------------------------")
# 输入2句话 句子长度为5， 嵌入维度为3
tensor_in = np.random.randint(low=0, high=3, size=30)
tensor_in = torch.from_numpy(tensor_in).float().view(2, 5, 3)
tensor_in[0][3:, ] = torch.zeros((2, 3), dtype=torch.float)
print("tensor_in:\n", tensor_in)

print("2.-------------------------- 倒排---------------------------------")
seq_lens = torch.tensor([3, 5], dtype=torch.int)
# 长度倒排对应的seq_len 的索引 1, 0
_, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
# 长度恢复对应的idx_sort的索引 1, 0
_, idx_unsort = torch.sort(idx_sort, dim=0)

# 长度重排结果
orded_seq_lengths = torch.index_select(seq_lens, dim=0, index=idx_sort)
# 输入重排结果
orded_tensor_in = torch.index_select(tensor_in, dim=0, index=idx_sort)
# 就是根据这两组索引将数据重新排列与恢复
print("长度重排:\n", orded_seq_lengths)
print("输入重排:\n", orded_tensor_in)

print("3.------------------------ 数据压缩 ------------------------------")
# 根据长度进行数据的重新组合，padidng的数据会被去掉
x_packed = nn_utils.rnn.pack_padded_sequence(orded_tensor_in, orded_seq_lengths, batch_first=True)

# input_size, hidden_size, n_layers
rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, nonlinearity='relu', bias=False)

# layers, batch_size, hidden_size, no matter if input is batch first
h0 = torch.randn((n_layers*batch_size, hidden_size)).view(n_layers, batch_size, hidden_size)

# 返回最后一层所有步的输出与最后一步所有层的输出
y_packed, h_n = rnn(x_packed, h0)
print('x_packed_data:\n', x_packed.data, "\nx_packed_batch_size\n", x_packed.batch_sizes)
print('y_packed_data:\n', y_packed.data, "\ny_packed_batch_size\n", y_packed.batch_sizes)

print("4.------------------------ 输出填充 ------------------------------")
y_sorted_padded, length = nn_utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
print("将y填充完整后是:\n", y_sorted_padded)
print(y_sorted_padded.shape)

print("5.------------------------ 输出恢复 ------------------------------")
right_y = torch.index_select(y_sorted_padded, dim=0, index=idx_unsort)
print("将y填充后再恢复文本顺序:\n", right_y)
print(right_y.shape)

print("6.------------------------ 隐层输出 ------------------------------")
# unsort output to original order
last_h = torch.index_select(h_n, dim=1, index=idx_unsort)
print("隐层最后一步的底层和顶层\n", last_h)

