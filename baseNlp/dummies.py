import pandas as pd
"""
离散特征的编码分为两种情况：
1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
https://blog.csdn.net/lujiandong1/article/details/52836051
"""
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'prize', 'class label']

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1}
df['size'] = df['size'].map(size_mapping)

class_mapping = {label: idx for idx, label in enumerate(set(df['class label']))}  # 不用枚举报错：too many values to unpack (expected 2)
df['class label'] = df['class label'].map(class_mapping)

print(pd.get_dummies(df))  # 既有编码，也有映射
print(pd.get_dummies(df.color))  # 部分列进行one-hot编码
