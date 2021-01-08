import numpy as np

# 视图view：是数据的引用，地址一致对一个改变， 另一个也会改变；
# 副本copy：对数据的完整拷贝，内存改变，操作副本不会影响原始数据。
print("********************************* view视图 *********************************")
# 1 直接赋值-->相当于视图
# 2 视图/浅拷贝
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.view()
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)
print("视图有新的地址")
print("修改视图的值,发现数据同步改变")
b[0] = [7, 8, 9]
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)
a[0] = [0, 0, 0]
print("修改原图的值,发现数据同步改变")
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)

# a.reshape((3, 2))  # 发现reshape函数不是inplace操作
a.shape = 3, 2
print("修改原图的shape,发现视图未改变")
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)
# 小结：view函数 数据不独立，但是shape独立。

print("********************************* 切片视图 *********************************")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a[1:]
c = a[1:]
print("切片后，地址为新地址")
print("a:", id(a[1:]), "\n", a[1:])
print("b:", id(b), "\n", b)
print("c:", id(c), "\n", c)

a[1] = [0, 0, 0]
b[-1] = [0, 0, 0]
print("切片后，原数组和切片的数据会相互影响")
print("a:", id(a[1:]), "\n", a[1:])
print("b:", id(b), "\n", b)
print("c:", id(c), "\n", c)
print("切片修改shape后，原数组和切片的数据仍然会相互影响！！！")
c.shape = 3, 2
c[0] = [100, 200]
print("a:", id(a[1:]), "\n", a[1:])
print("b:", id(b), "\n", b)
print("c:", id(c), "\n", c)

print("********************************* 副本/深拷贝 *********************************")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.copy()
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)

print("副本有新的地址，数据独立性")
b[0] = [7, 8, 9]
print("a:", id(a), "\n", a)
print("b:", id(b), "\n", b)

# 总结：
# 视图浅拷贝，shape独立，数据不独立，即便shape改变了，数据还是不独立。
# 与list的区别：对于切片操作，list会产生一个新的变量，数据独立，但是np的切片数据不具有独立性。
# list.deepcopy() == np.copy()
# list.copy() == np.view()


