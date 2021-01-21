import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
https://blog.csdn.net/tanlangqie/article/details/78656588
https://www.cnblogs.com/bugingcode/p/8310032.html
"""
# 从list中创建
# s1 = pd.Series([100, 23, 'bugingcode'])
# print(s1)
#
# # 内容和索引的设置
# s2 = pd.Series(np.random.randn(10), index=np.arange(1, 11))
# print(s2)
#
# # 从字典创建，索引可以是字符
# tempdict = {
#     'one': 'cat',
#     'two': 'dog',
# }
# s3 = pd.Series(tempdict)
# print(s3)
#
# s4 = pd.Series(np.random.randn(10), index=np.arange(1, 11))
# print(type(s4[1]))  # numpy.float64
# s4 = s4.cumsum()  # 元素求和
# print(s4)
# s4.plot()
# plt.show()


# dataframe 相当于二维数组，创建的时候指定数据、行索引、列索引，或者从csv文件创建
df = pd.DataFrame(np.random.randn(10, 6), index=range(1, 11), columns=list('ABCDEF'))
print(df)
# print(df['A'])  # series
# print('行数：{0}，列数：{1}'.format(len(df), df.columns.size))
# print('前3行3列：\n{}'.format(df.iloc[0:3, 0:3]))
# print('前3行3列：\n{}'.format(df.iloc[2:6, 2:5]))  # 任意切片索引从0开始，不是行列的具体值，是索引
# print('表示选取所有的行以及columns为A,C的列；'.format(df.loc[:, ['A', 'C']]))
a = df['A'].unique()
a = (a[a > 0])  # a>0返回每个位置的判断值（T or F），T的位置则输出
a = sorted(a)
print(a)
# print('表示选取数据集中大于1的数据\n{}'.format(df[df > 1]))
# print('表示选取数据集中A列大于1的行\n{}'.format(df[df.A > 0]))
#
# print('表示选取数据集中A列0--2的行\n{}'.format(df[df.A.isin(range(0, 2,))]))
#
# print('求每一列的平均值\n{}'.format(df.mean(0)))
# print('求每一行的平均值\n{}'.format(df.mean(1)))
#
# print('统计频数\n{}'.format(df.A.value_counts()))
# print('统计频数\n{}'.format(df['A'].value_counts()))  # 上式等价


