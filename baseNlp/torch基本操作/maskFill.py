import torch

a = torch.randn(2, 3, 4)
b = torch.arange(24).reshape(2, 3, 4).ge(18)
print("tensor a：\n", a)
print("tensor b：\n", b)
a = a.masked_fill(b, float("-inf"))
print("用 b mask a之后：\n", a)

print("最后softmax a：\n", torch.softmax(a, dim=2))


# tensor a：
#  tensor([[[-0.2248, -0.4389, -0.4409, -0.4439],
#          [ 0.2522, -0.3359, -1.7931, -1.4633],
#          [-0.3265,  1.4747, -2.3971, -0.0335]],
#
#         [[ 0.3025,  0.1693,  0.4256, -1.3716],
#          [ 2.4064, -0.2992,  0.1975, -2.2712],
#          [-0.0055, -0.7680,  0.5631,  1.1816]]])

# tensor b：
#  tensor([[[False, False, False, False],
#          [False, False, False, False],
#          [False, False, False, False]],
#
#         [[False, False, False, False],
#          [False, False,  True,  True],
#          [ True,  True,  True,  True]]])

# 用 b mask a之后：
#  tensor([[[-0.2248, -0.4389, -0.4409, -0.4439],
#          [ 0.2522, -0.3359, -1.7931, -1.4633],
#          [-0.3265,  1.4747, -2.3971, -0.0335]],
#
#         [[ 0.3025,  0.1693,  0.4256, -1.3716],
#          [ 2.4064, -0.2992,    -inf,    -inf],
#          [   -inf,    -inf,    -inf,    -inf]]])

# 最后softmax a：
#  tensor([[[0.2927, 0.2363, 0.2358, 0.2351],
#          [0.5363, 0.2979, 0.0694, 0.0965],
#          [0.1173, 0.7106, 0.0148, 0.1573]],
#
#         [[0.3131, 0.2741, 0.3541, 0.0587],
#          [0.9374, 0.0626, 0.0000, 0.0000],
#          [   nan,    nan,    nan,    nan]]])

# mask不是原地操作，需要再次复制，如果想避免nan的情况，
# 可以给mask的第二参数复制为一个很大的数
