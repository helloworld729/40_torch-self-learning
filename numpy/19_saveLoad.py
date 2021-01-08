import numpy as np

print("************************ 保存-save ************************")
# x = np.arange(1, 13).reshape(2, 6)
# np.save("20_example.npy", x)  # 保存文件名，数组

print("************************ 加载-load ************************")
y = np.load("20_example.npy")
print(type(y))
print(y)

