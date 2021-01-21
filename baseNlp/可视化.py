import numpy as np
import matplotlib.pyplot as plt
# ######################## 散点图 ######################################
# N = 100
# x1 = np.random.randn(N)
# y1 = np.random.randn(N)
# x2 = np.random.randn(N)
# y2 = np.random.randn(N)
#
# plt.scatter(x1, y1, color='g')
# plt.scatter(x2, y2, color='red')
# # plt.scatter(x, y, c=list(map(lambda num:int(num), x)))
#
# plt.legend(["case1", "case2"])
# plt.colorbar()
# plt.show()
# plt.close()

# ######################## 曲线图 ######################################
x = np.linspace(start=1, stop=5, num=400)
# y = x ** 2 + np.random.randn(400)
y = 1/x
plt.xlim(10, 450)
plt.plot(y, '-r')
plt.title("this is a title")
plt.legend(["loss_fuc"])
plt.text(x=100, y=0.7, s="hello world", color='black', size=20)
plt.show()
plt.close()

