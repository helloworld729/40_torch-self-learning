import numpy as np

a1 = np.empty((2, 3), dtype=int)    # 随机初始化
a2 = np.zeros((2, 3), dtype=float)  # 0初始化
a3 = np.ones((2, 3), dtype=float)   # 1初始化

# 随机创建正态分布等
a4 = np.array(np.random.randn(2, 3))
a5 = np.array(np.random.randint(0, 5, (2, 3)))  # 区间， 维度

# 从数据范围创建 创建
a6 = np.array(np.arange(10, 20+2, 2))
a7 = np.array(range(10, 20+2, 2))  # 和上一句等价
# 创建等差数列
c0 = np.linspace(start=1, stop=10, num=10, endpoint=True, retstep=False)
# 创建等比数列， base是底数，两个s是指数
c1 = np.logspace(start=1, stop=5, num=5, endpoint=True, base=10)

# 创建单位数组
a8 = np.eye(4)

b0 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# reshape
b1 = np.reshape(b0, (3, 4), order="C")  # 行优先
b2 = np.reshape(b0, (3, 4), order="F")  # 列优先

print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(a6)
print(a7)
print(a8)
print(c0)
print(c1)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(b1)
print(b2)

