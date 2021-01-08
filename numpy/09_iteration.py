import numpy as np

# 遍历与顺序控制
a = np.arange(6).reshape((2, 3))
for x in np.nditer(a, order="C"):
    print(x, end=" ")

print("\n")
for x in np.nditer(a, order="F"):
    print(x, end=" ")

# 开启读写模式/只读/只写
print("\n", "**************************************************")
for x in np.nditer(a, order="C", op_flags=["readwrite"]):
    x *= 2
print(a)

# ##############################################################
print("******************* 广播迭代 **************************")
a = np.arange(6).reshape((2, 3))
b = np.array([[1, 2, 3]])
for x, y in np.nditer([b, a]):
    print("{}：{}".format(x, y), end=" | ")

