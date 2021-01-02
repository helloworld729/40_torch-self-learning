# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2021/1/2 21:01
import os, json, random
from opencc import OpenCC

ts = OpenCC('t2s')  # 繁转简
st = OpenCC("s2t")  # 简转繁

# 接口测试
text = "一箭不中鵠，五湖歸釣魚。"
result = ts.convert(text)
print(result)
print(st.convert(result))

train_formmer = open("train.formmer", encoding="utf-8", mode="w")
train_latter  = open("train.latter",  encoding="utf-8", mode="w")
valid_formmer = open("valid.formmer", encoding="utf-8", mode="w")
valid_latter  = open("valid.latter",  encoding="utf-8", mode="w")
test_formmer  = open("test.formmer",  encoding="utf-8", mode="w")
test_latter   = open("test.latter",   encoding="utf-8", mode="w")

root_path = "./json/"
file_list = os.listdir(root_path)
for file_path in file_list:
    dir = root_path + file_path
    cont = json.load(open(dir, encoding="utf-8"))
    for one_poem in cont:
        NUM = random.random()
        if NUM < 0.8:
            formmer_writer = train_formmer
            latter_writer  = train_latter
        elif NUM < 0.9:
            formmer_writer = valid_formmer
            latter_writer  = valid_latter
        else:
            formmer_writer = test_formmer
            latter_writer  = test_latter

        # 所有句 构成的一维列表
        paragraphs = one_poem["paragraphs"]

        for pair in paragraphs:
            try:
                if "？"in pair:
                    pair = pair.split("？")
                    formmer = pair[0]
                    formmer = ts.convert(formmer) + "？"
                else:
                    pair = pair.split("，")
                    formmer = pair[0]
                    formmer = ts.convert(formmer) + "，"

                latter = pair[1]
                latter = ts.convert(latter)
                if len(formmer) == len(latter):
                    formmer_writer.write(formmer + "\n")
                    latter_writer.write(latter + "\n")
            except:
                print(pair)

