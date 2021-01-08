import numpy as np
a = np.array([2, 4, 6, 8])
b = np.array([3, 5, 7, 9])
print("操作数1：", a)
print("操作数2：", b)
print("*********************************** 加减乘除 *********************************** ")
print("a+b：", np.add(a, b))
print("a-b：", np.subtract(a, b))
print("a*b：", np.multiply(a, b))
print("a/b：", np.divide(a, b))

print("*********************************** 求幂取余 *********************************** ")
print("操作数1的平方：", np.power(a, 2))
print("操作数1除以2取余：", np.mod(a, 2))

