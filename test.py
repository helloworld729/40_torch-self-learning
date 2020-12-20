# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2020/12/18 14:55
import torch
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
b.add_(1)
print(a)
print(b)
print(b)