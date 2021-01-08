import numpy as np
print("*********************** numpy.sort-排序 ***********************")
# numpy.sort(a, axis, kind, order)
# kind 可以选择："quicksort"、 "mergesort"、"heapsort"
# 容易搞混的地方：假如我们想把每一行的元素排序，那么axis
# 需要设定为1，按照类分割后，将列分的元素排序
a = np.array([[1, 3, 2], [6, 5, 4]])
b = np.sort(a,axis=1)
print("原数组: \n", a)
print("排序后: \n", b)

print("*********************** numpy.argsort-顺序索引 ***********************")
# numpy.argsort() 返回数值从小到大的索引值
c = np.argsort(a)
print("原数组: \n", a)
print("原数组有小到大的索引值: \n", c)

print("*********************** numpy.argmax()-最大值索引 ***********************")
d0 = np.argmax(a, axis=1)
d1 = np.argmin(a, axis=1)
print("原数组: \n", a)
print("argmax-index: \n", d0)
print("argmin-index: \n", d1)

print("*********************** numpy.where 筛选 ***********************")
x = np.arange(1, 10).reshape((3, 3))
y = np.where(x > 3)
print("原数组: \n", x)
print("筛选X>3: \n", y)
print(x[y])


