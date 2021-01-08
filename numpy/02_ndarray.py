import numpy as np
# print("版本号：", np.__version__)
# N维数组的定义 数据+类型
a = np.array([1, 2, 3])                    # 一维数组
b = np.array([[1, 2], [3, 4]])             # 二维数组
d = np.array([1+2j, 3+4j], dtype=complex)  # 复数数组
print("一维数组 a", a)
print("二维数组 b", b)
print("复数数组 d", d)

