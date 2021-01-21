from sklearn.datasets import load_iris
data = load_iris()
labels = data.target[[48, 49, 50]]  # 求出48,49,50样本的 label值
print(labels)
print(list(data.target_names))  # 打印所有的标签名字
