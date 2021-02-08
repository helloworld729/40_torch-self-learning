# 交叉熵、KL散度损失
# 先实例化，再调用

import torch
import torch.nn as nn
from torch.nn import KLDivLoss

# 线型神经元输出模拟
# 交叉熵损失函数
label = torch.tensor([1.0, 0, 0])

criterier1 = KLDivLoss(reduction="mean")  # log_softmax

# 概率模拟
for p in [[0.8, 0.1, 0.1], [0.9, 0.05, 0.05], [1.0, 0.0, 0]]:
    probility = torch.tensor(p).reshape(1, 3)
    logProbility = torch.log(probility)
    criterier4 = KLDivLoss()
    loss4 = criterier4(logProbility, label)
    print(loss4)

# tensor(0.2231)
# tensor(0.1054)
# tensor(0.)

# tensor(0.0744)
# tensor(0.0351)
# tensor(0.)

# 上面是交叉熵，下面是相对熵求了平均(除以3)，发现当上为0的时候，相对熵确实是交叉熵。
# 所以对于硬分类器，没有必要考虑相对熵，相对熵更适合平滑后的label或者本身label就不是one-hot
