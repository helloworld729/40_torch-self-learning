# -*- coding:utf-8 -*-
import torch
import os
path = "../../dataBert/"
# path = "../../bert_data/"
ptFile = os.listdir(path)
for f in ptFile:
    data = torch.load(path + f)
    for d in data:
        for k, v in d.items():
            print(k, ": ", v)
        print("+++++++++++++++++++++++++++++++++++++++++++++")

#
# for f in ptFile:
#     data = torch.load(path + f)
#     for d in data[:2]:
#         for k, v in d.items():
#             print(k, ": ", v)
#         print("+++++++++++++++++++++++++++++++++++++++++++++")