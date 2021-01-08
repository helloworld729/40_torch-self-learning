# d-type即数据类型对象，用来描述与数组对应的区域如何使用，具体
# 包括以下各个方面：数据类型(可以是结构化类型(名称，类型))、大小、顺序(大端还是小端)
# 构造语法为：numpy.dtype(object, align, copy)

# 关于数据类型，numpy有很多的数据类型，而且每一个内置类型都有简写的字母与之对应
# 例如 i表示int，f表示float， U表示unicode
# 结构化数据类型 类似c语言中的结构体，输出时可以指定输出其中的一项

import numpy as np
dt1 = np.dtype(np.int32)  # 掌握这种易读型即可
dt2 = np.dtype("f")       # 等价于np.float
dt3 = np.dtype("<i4")

# 定义结构体数据类型(元组列表)并应用到array对象
dt4 = np.dtype([("name", "U20"), ("age", "i1"), ("marks", "f4")])  # name，type, U 是UNICODE的意思
a = np.array([("Knight", 25, 96), ("Nancy", 23, 95)], dtype=dt4)
print(dt1)
print(dt2)
print(dt3)
print(a)
print(a["name"])

