import numpy as np

print("************************************* 1 三角函数 *************************************")
angle = np.array([30, 45, 60, 90])  # 角度值
print("origin angle are", angle)
# 面向弧度值
print("after sin function", np.sin(np.pi *angle/180, ))
print("after cos function", np.cos(np.pi *angle/180, ))
print("after tan function", np.tan(np.pi *angle/180, ))

print("************************************* 2 舍入函数 *************************************")
# np.around(a, decimal)
a = np.array([0.45, 0.55, 0.65])
print("原数组：", a)
print("舍入后：", np.around(a, 1))
# 结果是：
# [0.45 0.55 0.65]
# [0.4  0.6  0.6]
# 解释：4舍6入，5到偶，假如一个数刚好以5结尾，那么摄入到最近的偶数所以0.45的结果为0.4

print("************************************ 3 floor函数 *************************************")
# 无论正负，向数轴左侧取整
a = np.array([-1.6, -0.2, -0.6, 0.6, 1.3])
print("原数组：", a)
print("舍入后：", np.floor(a))

print("************************************ 4 ceil函数 *************************************")
# 无论正负，向数轴右侧取整
a = np.array([-1.6, -0.2, -0.6, 0.6, 1.3])
print("原数组：", a)
print("舍入后：", np.ceil(a))

