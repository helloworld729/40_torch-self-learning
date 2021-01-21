from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
from scipy.stats import multivariate_normal
# import numpy as np
# np.random.dirichlet()
samplesource = multivariate_normal(mean=[5,-1], cov=[[1,1],[1,4]])  # 二维正态分布

def p_ygiven_by_x(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt((1 - rho ** 2) * (s2**2))))

def p_xgiven_by_y(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt((1 - rho ** 2) * (s1**2))))

N = 5000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5   # 均值1
m2 = -1  # 均值2
s1 = 1  # 方差1
s2 = 2  # 方差2

rho = 0.5
y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiven_by_y(y, m1, m2, s1, s2)
        y = p_ygiven_by_x(x, m1, m2, s1, s2)
        z = samplesource.pdf([x,y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o')
plt.show()