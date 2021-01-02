# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/1/2 21:01
import os, json
from opencc import OpenCC

ts = OpenCC('t2s')  # 繁转简
st = OpenCC("s2t")  # 简转繁

# 接口测试
# text = "一箭不中鵠，五湖歸釣魚。"
# result = ts.convert(text)
# print(result)
# print(st.convert(result))

path = "./json"
file_list = os.listdir(path)
for file_path in file_list:
    cont = json.loads(file_path)
    print(cont)
    print(type(cont))
    break


