import torch
import numpy as np
import torch.nn as nn
from torch.nn import utils as nn_utils

batch_size  = 2
hidden_size = 5
# embedding dimension
input_size = 3
n_layers   = 4

print("1.-------------------------- 数据 --------------------------------")
# 输入2句话 句子长度为5， 嵌入维度为3
tensor_in = torch.randn(2, 5, 3)
tensor_in[0][3:, ] = torch.zeros((2, 3), dtype=torch.float)
print("tensor_in:\n", tensor_in.data.numpy())
# tensor_in:
#  [[[-0.16868705 -1.4781348  -0.06997177]
#   [ 1.0002288   2.2042887   1.1671765 ]
#   [ 1.5773681   0.01774285 -0.62640065]
#   [ 0.          0.          0.        ]
#   [ 0.          0.          0.        ]]
#
#  [[-0.45969158  0.32685533 -0.15799478]
#   [-0.3737488  -0.5351009   0.4423607 ]
#   [-0.44480538  0.34696054  0.38751316]
#   [ 1.0240709  -0.7715575   0.804514  ]
#   [-0.3817314  -0.7076132   1.5074275 ]]]

print("2.-------------------------- 倒排---------------------------------")
seq_lens = torch.tensor([3, 5], dtype=torch.int)
# 长度倒排对应的seq_len 的索引 1, 0
_, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
# 长度恢复对应的idx_sort的索引 1, 0
_, idx_unsort = torch.sort(idx_sort, dim=0)

# 长度重排结果
orded_seq_lengths = seq_lens.index_select(dim=0, index=idx_sort)
# 输入重排结果
orded_tensor_in = tensor_in.index_select(dim=0, index=idx_sort)
# 就是根据这两组索引将数据重新排列与恢复
print("长度重排:\n", orded_seq_lengths.data.numpy())
print("输入重排:\n", orded_tensor_in.data.numpy())
# 长度重排:
#  [5 3]
# 输入重排:
#  [[[-0.45969158  0.32685533 -0.15799478]
#   [-0.3737488  -0.5351009   0.4423607 ]
#   [-0.44480538  0.34696054  0.38751316]
#   [ 1.0240709  -0.7715575   0.804514  ]
#   [-0.3817314  -0.7076132   1.5074275 ]]
#
#  [[-0.16868705 -1.4781348  -0.06997177]
#   [ 1.0002288   2.2042887   1.1671765 ]
#   [ 1.5773681   0.01774285 -0.62640065]
#   [ 0.          0.          0.        ]
#   [ 0.          0.          0.        ]]]

print("3.------------------------ 数据压缩 ------------------------------")
# 根据长度进行数据的重新组合，padidng的数据会被去掉
x_packed = nn_utils.rnn.pack_padded_sequence(orded_tensor_in, orded_seq_lengths, batch_first=True)

# input_size, hidden_size, n_layers
rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True, nonlinearity='relu', bias=False)

# layers, batch_size, hidden_size, no matter if input is batch first
h0 = torch.randn((n_layers*batch_size, hidden_size)).view(n_layers, batch_size, hidden_size)

# 返回最后一层所有步的输出与最后一步所有层的输出
y_packed, h_n = rnn(x_packed, h0)
print('x_packed_data:\n', x_packed.data.data.numpy(), "\nx_packed_batch_size\n", x_packed.batch_sizes.data.numpy())
print('y_packed_data:\n', y_packed.data.data.numpy(), "\ny_packed_batch_size\n", y_packed.batch_sizes.data.numpy())
# x_packed_data:
#  [[-0.45969158  0.32685533 -0.15799478]
#  [-0.16868705 -1.4781348  -0.06997177]
#  [-0.3737488  -0.5351009   0.4423607 ]
#  [ 1.0002288   2.2042887   1.1671765 ]
#  [-0.44480538  0.34696054  0.38751316]
#  [ 1.5773681   0.01774285 -0.62640065]
#  [ 1.0240709  -0.7715575   0.804514  ]
#  [-0.3817314  -0.7076132   1.5074275 ]]
# x_packed_batch_size
#  [2 2 2 1 1]
# y_packed_data:
#  [[0.1647667  0.         0.         0.50981945 0.        ]
#  [0.         0.         0.72543794 0.05721357 0.3135454 ]
#  [0.         0.07811449 0.2834246  0.09436507 0.        ]
#  [0.         0.         0.         0.         0.0664663 ]
#  [0.         0.         0.         0.         0.0845597 ]
#  [0.13841702 0.00381942 0.12228005 0.         0.        ]
#  [0.16836245 0.06634697 0.23988909 0.         0.        ]
#  [0.         0.         0.37418118 0.         0.        ]]
# y_packed_batch_size
#  [2 2 2 1 1]

print("4.------------------------ 输出填充 ------------------------------")
y_sorted_padded, length = nn_utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
print("将y填充完整后是:\n", y_sorted_padded.data.numpy())
print(y_sorted_padded.shape)
# 将y填充完整后是:
#  [[[0.1647667  0.         0.         0.50981945 0.        ]
#   [0.         0.07811449 0.2834246  0.09436507 0.        ]
#   [0.         0.         0.         0.         0.0845597 ]
#   [0.16836245 0.06634697 0.23988909 0.         0.        ]
#   [0.         0.         0.37418118 0.         0.        ]]
#
#  [[0.         0.         0.72543794 0.05721357 0.3135454 ]
#   [0.         0.         0.         0.         0.0664663 ]
#   [0.13841702 0.00381942 0.12228005 0.         0.        ]
#   [0.         0.         0.         0.         0.        ]
#   [0.         0.         0.         0.         0.        ]]]
# torch.Size([2, 5, 5])

print("5.------------------------ 输出恢复 ------------------------------")
right_y = torch.index_select(y_sorted_padded, dim=0, index=idx_unsort)
print("将y填充后再恢复文本顺序:\n", right_y.data.numpy())
print(right_y.shape)
# 将y填充后再恢复文本顺序:
#  [[[0.         0.         0.72543794 0.05721357 0.3135454 ]
#   [0.         0.         0.         0.         0.0664663 ]
#   [0.13841702 0.00381942 0.12228005 0.         0.        ]
#   [0.         0.         0.         0.         0.        ]
#   [0.         0.         0.         0.         0.        ]]
#
#  [[0.1647667  0.         0.         0.50981945 0.        ]
#   [0.         0.07811449 0.2834246  0.09436507 0.        ]
#   [0.         0.         0.         0.         0.0845597 ]
#   [0.16836245 0.06634697 0.23988909 0.         0.        ]
#   [0.         0.         0.37418118 0.         0.        ]]]
# torch.Size([2, 5, 5])

print("6.------------------------ 隐层输出 ------------------------------")
# unsort output to original order
last_h = torch.index_select(h_n, dim=1, index=idx_unsort)
print("隐层最后一步的底层和顶层\n", last_h.data.numpy())
# 隐层最后一步的底层和顶层
#  [[[0.14286333 0.03621571 0.         0.23367456 0.        ]
#   [0.         0.         0.2327425  0.96003914 0.72193015]]
#
#  [[0.13841702 0.00381942 0.12228005 0.         0.        ]
#   [0.         0.         0.37418118 0.         0.        ]]]

