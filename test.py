# -*- coding:utf-8 -*-
# Author:Knight
# @Time:2020/12/18 14:55

lst = ["a", "b", "c"]

# a = [d for d in lst]

def nextTest():
    a = (d for d in lst)
    while True:
        while a is not None:
            print(next(a))

nextTest()