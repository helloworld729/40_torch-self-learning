import torch
import torch.nn.functional as F
print(torch.sigmoid(torch.tensor([1.])))
# print(F.sigmoid(torch.tensor([1.])))
# nn.functional.sigmoid is deprecated. Use torch.sigmoid instead

x = torch.arange(12).reshape(3, 4).float()
print(x)
print(torch.sigmoid(x))
print(torch.softmax(x, dim=0))
print(torch.softmax(x, dim=1))

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# tensor([[0.5000, 0.7311, 0.8808, 0.9526],
#         [0.9820, 0.9933, 0.9975, 0.9991],
#         [0.9997, 0.9999, 1.0000, 1.0000]])

# tensor([[3.2932e-04, 3.2932e-04, 3.2932e-04, 3.2932e-04],
#         [1.7980e-02, 1.7980e-02, 1.7980e-02, 1.7980e-02],
#         [9.8169e-01, 9.8169e-01, 9.8169e-01, 9.8169e-01]])

# tensor([[0.0321, 0.0871, 0.2369, 0.6439],
#         [0.0321, 0.0871, 0.2369, 0.6439],
#         [0.0321, 0.0871, 0.2369, 0.6439]])

# 小结：sigmoid面向单个元素
# softmax 需要设置维度

