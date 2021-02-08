
# ################################## 关于矩阵contiguous ##################################
# 预备知识： storage、tensor、size、stride。
# pytorch中的storage指的是连续的内存块，而tensor则是映射到storage的视图，他把单条的内存区域映射成了n维的空间视图。
# size是tensor的维度，storage offset是数据在storage中的索引，stride是storage中对应于tensor的相邻维度间第一个索引的跨度。示例如下：
# 1 2 3
# 4 5 6
# 上面的tensor的size是(2， 3)，storage为 1 2 3 4 5 6(行优先)
# stride为(3，1)表示分别表示在第0维和第1维上到达相邻元素的跨度
import torch
x7 = torch.arange(12).reshape(3, 4)
print(x7)
x8 = x7.transpose(0, 1)
print(x8)  # 等价于x7.transpose(1, 0)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# tensor([[ 0,  4,  8],
#         [ 1,  5,  9],
#         [ 2,  6, 10],
#         [ 3,  7, 11]])

print(id(x7), x7.data_ptr())
print(id(x8), x8.data_ptr())
# 43081584144 43261304192
# 43081638824 43261304192

print(x7.transpose(0, 1).view(1, 12))
# view size is not compatible with input tensor's size and stride
# (at least one dimension spans across two contiguous subspaces).
# Use .reshape(...) instead.

print(x7.transpose(0, 1).contiguous().view(1, 12))
# tensor([[ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11]])

# 小结：view操作需要contiguous()操作。reshape()函数，相当于tensor.contiguous().view()
# transpose、permute 操作不修改底层一维数组。但是新建了一份Tensor元信息，并在新的元信息中的 重新指定 stride。
# 操作前后，tensor地址改变，但是数据指针指向的位置相同，即前文的id与data_ptr
# torch.view 方法约定了不修改数组本身，只是使用新的形状查看数据。像transpose、permute这种修改数组映射的操作
# 在使用后不能直接使用view函数

# ！！！所以，如果不关注内存是否会改变可以使用reshape()函数，要求内存不改变可以考虑使用view操作。
# 优先使用view函数

