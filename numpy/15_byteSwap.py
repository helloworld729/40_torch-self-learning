import numpy as np
print("********************************* np.byteswap Function *********************************")
# a = np.array([1, 256, 8755], dtype=np.int16)
# print(a)
# print(map(hex, a))
# a.byteswap(True)
# print(a)
# print(map(hex, a))

import numpy as np

a = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)
print('我们的数组是：')
print(a)
print('以十六进制表示内存中的数据：')
print(list(map(bin, a)))
# byteswap() 函数通过传入 true 来原地交换
print('调用 byteswap() 函数：')
print(a.byteswap(True))
print('十六进制形式：')
print(list(map(bin, a)))
# 我们可以看到字节已经交换了

# 以600为例，600的二进制是 0000 0010 0101 1000，一个字节是8位 那么交换后就是 0101 1000 0000 0010-->101100000000010

