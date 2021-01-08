import numpy as np
import numpy.matlib

print("******************************* 矩阵创建 *******************************")
print("随机初始化的 empty矩阵")
a = np.matlib.empty((2, 3))
print(type(a))
print(a)

print("zeros矩阵")
print(np.matlib.zeros((2, 3)))

print("ones矩阵")
print(np.matlib.ones((2, 3)))

print("对角矩阵")
# 参数分别是行、列、1放置的索引
print(np.matlib.eye(2, 3, 0, int))
print(np.matlib.eye(2, 3, 1))

print("identity 单位矩阵")
print(np.matlib.identity(2))

print("rand 随机矩阵")
print(np.matlib.rand(2, 3))

print("数组-矩阵 互相转化, 浅拷贝")
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.asmatrix(x)  # 不会创建副本
x[0][0] = 0
print(type(y))
print(y)

print("修改矩阵元素")
# y[0][0] = 1  # not work
y[0, 0] = 1
z = np.asarray(y)
print(type(z))
print(z)

