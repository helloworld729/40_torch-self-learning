import torch
import numpy as np
import torch.nn as nn
# ###################################  NP模拟 #########################################
# input_numpy = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
# input = torch.from_numpy(input_numpy)  # 2  3
# m = nn.LayerNorm(3)
# output = m(input)
# input_mean = np.mean(input_numpy, axis=1, keepdims=True)
# input_std = np.std(input_numpy, axis=1, keepdims=True)
# print(input_mean.squeeze())
# print(input_std.squeeze())
# print((input_numpy - input_mean) / input_std)
# print(output.data.numpy())

# [2. 5.]
# [0.8164966 0.8164966]
# [[-1.2247448  0.         1.2247448]
#  [-1.2247448  0.         1.2247448]]

# [[-1.2247354  0.         1.2247355]
#  [-1.2247343  0.         1.2247348]]

# ######################################## Torch模拟 ###########################################
# input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)  # 2  3
m = nn.LayerNorm(3)
# output = m(input)
# input_mean = torch.mean(input, dim=1, keepdim=True)
# input_std = torch.std(input, dim=1, keepdim=True, unbiased=False)
# print("均值squeeze：", input_mean.data.numpy().squeeze())
# print("标准方差：", input_std.data.numpy().squeeze())
# print("手工计算LayerNorm：\n", ((input - input_mean) / input_std).data.numpy())
# print("接口计算LayerNorm：\n", output.data.numpy())
# 不能在需要计算梯度(requires_grad=True)的Tensor上直接调用numpy()函数，所以要么detach，
# 要么.data,虽然.data仍然是张量，但是requires_grad属性为False

# 均值squeeze： [2. 5.]
# 标准方差： [0.8164966 0.8164966]
# 手工计算LayerNorm：
#  [[-1.2247448  0.         1.2247448]
#  [-1.2247448  0.         1.2247448]]
# 接口计算LayerNorm：
#  [[-1.2247354  0.         1.2247355]
#  [-1.2247343  0.         1.2247348]]

# 总结：以文本为例，我们每一个单词后会有embedding_size，LayerNorm不关注其他的属性，因为他是在一个
# word的范围内归一化，例如某个单词的词向量为[1,2,3]那么均值为2， 方差为0.8164966
# 归一化后为[-1.2247448  0. 1.2247448]这就是用每一个element x通过x-mean/std计算而来

# ######################################## 对比 ###########################################
# input_numpy = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
# input_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)  # 2  3

# numpy_std = np.std(input_numpy, axis=1)
# torch_std = torch.std(input_torch, dim=1, unbiased=False)        # 不加unbiased会得到不想要的结果
# torch_std_unbias = torch.std(input_torch, dim=1, unbiased=True)  # 不加unbiased会得到不想要的结果
#
# print('numpy方差：     {}'.format(numpy_std))
# print('torch方差：     {}'.format(torch_std))
# print('torch修正方差： {}'.format(torch_std_unbias))
# numpy方差：     [0.8164966 0.8164966]
# torch方差：     tensor([0.8165, 0.8165])
# torch修正方差： tensor([1., 1.])

# torch求方差，要注意unbiased属性，一般需要设置为False

