import numpy as np
a = np.array([[0, 3, 2], [4, 5, 6], [9, 8, 7]])
# 设定axis=0，实际就按照你列比较
print("*********************************** axis max value ***********************************")
print("origin array:\n", a)
print("全局最大值：", np.amax(a))
print("最大的行", np.amax(a, 0))
# 这个函数 可以用where筛选，控制方法见函数定义。

print("*********************************** 沿轴最大差值  ***********************************")
print(np.ptp(a, axis=0))

print("***********************************  沿轴中位数  ***********************************")
print("中位数：", np.median(a, axis=0))
print("算数平均值：", np.mean(a, axis=0))
print("加权平均值：", np.average(a, axis=0, weights=[4,2,2]))

print("******************************** 小于观察值的百分比  ***********************************")
point = 50
print("约有{}%的数小于:".format(point), np.percentile(a, point)) # 第二个参数是设定的百分位数

print("************************************** 标准差  *****************************************")
print("全局标准差:",np.std(a))
print("全局方差：", np.var(a))
print("axis=0方差", np.var(a, axis=0))

