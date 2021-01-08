import numpy as np
a = np.array([[1, 2+4j], [2+3j, 4+6j]])
print("维度：", a.shape)
print("元素总个数：", a.size)
print("元素类型：", a.dtype)
print("每个元素的字节数：", a.itemsize)
# print("内存信息：", a.flags)
print("实部：", a.real)
print("虚部：", a.imag)

# ######################## 关于reshape #############################
print("###################### 关于reshape #########################")
b = a.reshape((4,))
b[0] = 7+8j
print("#在reshape后的对象中，改变元素内容，原来array的内容也会改变#")
print(a)

