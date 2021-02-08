# 交叉熵、KL散度损失
# 先实例化，再调用

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 线型神经元输出模拟
# 交叉熵损失函数
label = torch.tensor([0])
outputs = torch.tensor([-0.7, 0.2, 0.8]).reshape(1, 3)
criterier1 = CrossEntropyLoss()  # log_softmax
loss1 = criterier1(outputs, label)
print(loss1)

# 负对数似然函数
lsm = nn.LogSoftmax(dim=1)  # 激活函数
criterier2 = nn.NLLLoss()   # 损失函数
loss2 = criterier2(lsm(outputs), label)  # 求损失
print(loss2)

# 手动模拟
soft = nn.Softmax(dim=-1)
probility = soft(outputs)
logProbility = torch.log(probility)
criterier3 = nn.NLLLoss()
loss3 = criterier3(logProbility, label)
print(loss3)
print(torch.log(torch.tensor([2.71828])))  # 自然对数

# 实验表明，torch中的log对数是自然底数e

# tensor(0.9398)
# tensor(0.9398)
# tensor(0.9398)
# tensor([1.0000])

# https://zhuanlan.zhihu.com/p/98785902

# 概率模拟
for p in [[0.8, 0.1, 0.1], [0.9, 0.05, 0.05], [1.0, 1.0, 0]]:
    probility = torch.tensor(p).reshape(1, 3)
    logProbility = torch.log(probility)
    criterier4 = nn.NLLLoss()
    loss4 = criterier4(logProbility, label)
    print(loss4)

# tensor(0.2231)
# tensor(0.1054)
# tensor(0.)

