import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target  # X [1083, 64]  ||y [1083]
n_samples, n_features = X.shape  # [1083, 64]

# 相当于1083张 8*8 的图片数据

'''显示原始数据'''
n = 20  # 每行20个数字，每列20个数字
img = np.zeros((10 * n, 10 * n))  # 10*10的空间，8*8的数据，剩下的位置默认为0，更好的分开
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()

'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)  # 1083, 2

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化  所有的数据
plt.figure(figsize=(8, 8))
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
plt.colorbar()

flag_dict = {digit: 0 for digit in range(6)}
pass_step = 0

for i in range(X_norm.shape[0]):  # shape[0]表示点的个数
    # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),  # x值，y值，标签值，颜色，字体
    #          fontdict={'weight': 'bold', 'size': 9})
    if flag_dict[y[i]] == 0 and pass_step == 10:
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color='black',  # x值，y值，标签值，颜色，字体
                 fontdict={'weight': 'bold', 'size': 39})
        flag_dict[y[i]] = 1
        pass_step = 0
    elif flag_dict[y[i]] == 0:
        pass_step += 1


# plt.xticks([])  # 坐标刻度设置
# plt.yticks([])  # 坐标刻度设置
# plt.show()
plt.savefig("T-sne.png")
plt.close()

