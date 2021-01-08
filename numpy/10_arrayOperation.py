import numpy as np

print("************************ 修改形状 *************************")
# 修改形状 reshape
print("------------------------- flat迭代 -------------------------")
a0 = np.arange(1, 10).reshape(3,3)
print("原数组：\n", a0)
for x in a0.flat:  # flat 是类似于nditer的迭代器，但是不能原地修改
    print(x, end=" ")
print()
print("------------------------- flatten压平 -------------------------")
# 压平操作 flatten()|ravel()
print(a0.flatten(), "flatten shape is", a0.flatten().shape)
print(a0.ravel(), "ravel shape is", a0.ravel().shape)
# the difference between flatten and ravel is ravel后修改元素
# 原数组的内容也会跟这变化，flatten相当于深拷贝了

# 翻转数组-转置.T
print("数组转置\n", a0.T)
# 其他轴操作 rollaxis、swapaxes

print("************************ 修改数组维度*************************")
# 修改数组维度
a = np.arange(4).reshape(1, 4)

print('原数组：\n', a)
# board-cast 返回只读视图
print('调用 broadcast_to 函数之后：')
print(np.broadcast_to(a, (4, 4)))

# expand-dims 维度提升-在指定位置插入axis
d0 = np.arange(1,5).reshape((2, 2))
print("d维度扩展之前：shape", d0.shape)
print(d0)
d1 = np.expand_dims(d0, 0)
print("d维度扩展之后：shape", d1.shape)
print(d1)
print("************************** squeeze ***************************")
# squeeze -删除维度为1轴, 当有多个1维的axis可以指定squeeze的位置，否则全部清除
print("维度挤压之后\n", d1.squeeze().shape)


print("************************** 数组连接 ***************************")
print("--------------------------concatenate-------------------------")
b1 = np.array([[1, 2], [3, 4]])
b2 = np.array([[5, 6], [7, 8]])
print("连接之前的b1\n", b1)
print("连接之前的b1\n", b2)
print("按照行连接\n", np.concatenate((b1, b2), 0))
print("按照列连接\n", np.concatenate((b1, b2), 1))

print("-----------------------------stack----------------------------")
# https://blog.csdn.net/wgx571859177/article/details/80987459
c1 = np.array([[1, 2, 3], [4, 5, 6]])
c2 = np.array([[7, 8, 9], [10, 11, 12]])
print("按照第一维度谅解\n", np.stack((c1, c2), 0))
print("按照第二维度谅解\n", np.stack((c1, c2), 1))
print("按照第三维度谅解\n", np.stack((c1, c2), 2))

print("************************** 数组分割 ***************************")
print("--------------------------- split ----------------------------")
d1 = np.arange(9)
print(np.split(d1, 3))
print(np.split(d1, [1, 4, 7]))  # 切点属于右侧区间
# 还有hsplit和vsolit分别为水平和垂直分割

print("************************** 数组resize ***************************")
e1 = np.arange(1, 13).reshape((2, 6))
print(np.resize(e1, (3, 4)))
# 如果容量小会截断，容量大会重复

print("************************** 数组append ***************************")
f1 = np.array([[1, 2, 3], [4, 5, 6]])
print("拼接到下边：\n", np.append(f1, [[7, 8, 9]], axis=0))  # careful the dimension
print("拼接到右边：\n", np.append(f1, [[4, 5], [7, 8]], axis=1))  #

print("************************** 数组insert ***************************")
# Numpy.insert(数组, 位置, 数据, 轴)
print("依据行插入\n", np.insert(f1, 1, [7, 8, 9], axis=0))
print("依据列插入\n", np.insert(f1, 3, [4, 7], axis=1))
print("先展开再插入\n", np.insert(f1, 0, [5]))

print("************************** 数组delete ***************************")
# Numpy.delete(数组, 位置, 轴)
f2 = np.insert(f1, 2, [7, 8, 9], axis=0)
print("删除之前\n", f2)
print("删除之后\n", np.delete(f2, 2, axis=0))

print("************************** 数组unique ***************************")
# numpy.unique(arr, return_index, return_inverse, return_counts)
# 新在旧的位置，就在新的位置，去重数组元素出现的次数
f3 = np.delete(f2, 2, axis=0)
f4 = np.insert(f3, 2, [1,2,3], axis=0)
print(f4)
info = np.unique(f4, True, True, True)  # 全为True返回4个数组，全为False只返回去重结果
print(info)

