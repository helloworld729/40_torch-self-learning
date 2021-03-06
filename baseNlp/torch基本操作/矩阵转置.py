import torch
# ################################## 关于矩阵转置 ##################################
x9 = torch.arange(24).reshape(2, 3, 4)
print(x9)
x10 = x9.transpose(0, 1)  # 3 2 4
print(x10)
x11 = x10.permute(1, 0, 2)  # 2, 3, 4
print(x11)
x11[0][0][0] = 100
print(x11)
print(x10)
print(x9)

# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
# tensor([[[ 0,  1,  2,  3],
#          [12, 13, 14, 15]],
#
#         [[ 4,  5,  6,  7],
#          [16, 17, 18, 19]],
#
#         [[ 8,  9, 10, 11],
#          [20, 21, 22, 23]]])
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
# tensor([[[100,   1,   2,   3],
#          [  4,   5,   6,   7],
#          [  8,   9,  10,  11]],
#
#         [[ 12,  13,  14,  15],
#          [ 16,  17,  18,  19],
#          [ 20,  21,  22,  23]]])
# tensor([[[100,   1,   2,   3],
#          [ 12,  13,  14,  15]],
#
#         [[  4,   5,   6,   7],
#          [ 16,  17,  18,  19]],
#
#         [[  8,   9,  10,  11],
#          [ 20,  21,  22,  23]]])
# tensor([[[100,   1,   2,   3],
#          [  4,   5,   6,   7],
#          [  8,   9,  10,  11]],
#
#         [[ 12,  13,  14,  15],
#          [ 16,  17,  18,  19],
#          [ 20,  21,  22,  23]]])

import time
t1 = time.time()
for i in range(100000):
    x9.transpose(0, 2)
t2 = time.time()
for i in range(100000):
    x9.permute(2, 1, 0)
t3 = time.time()
#
print(t2 - t1)
print(t3 - t2)
# 1.2031407356262207
# 1.4375255107879639

# transpose、permute不改变底层数据的存储，所以内存是共享的，一个变都变
# 小结：transpose 只能调整2个维度，而permute可以任意数目的维度
# 优先transpose

