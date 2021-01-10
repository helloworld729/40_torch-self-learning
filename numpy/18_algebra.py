import numpy as np

print("****************************** dot ******************************")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(a, b))
print("************* 测试三维数组相乘 *************")
a = np.arange(1, 13).reshape((2, 2, 3))
b = np.array([[0, 0], [1, 1], [2, 2]])
# print(a, "\n", b)
print(np.dot(a, b))
# 一个2*2*3的矩阵和一个3*2的矩阵相乘，生成一个2*2*2的矩阵，
# 可以看做两个2*3的矩阵分别和3*2的矩阵相乘，生成两个2*2的矩阵
# 然后把结果拼接在一块。

print("****************************** det-行列式 ******************************")
x = np.arange(1, 5).reshape((2, 2))
print("应该是-2，实际上是：{0:.2f}".format(np.linalg.det(x)))

print("****************************** inv-逆矩阵 ******************************")
print(np.linalg.inv(x))

print("****************************** solve-解方程 ******************************")
x = np.arange(1, 5).reshape(2, 2)
y = np.array([1, 8]).reshape((2, 1))
print(np.linalg.solve(x, y))
