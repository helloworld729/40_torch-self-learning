data_base = ["Knight", 24, "13912968612", "18561653118"]

name, age, *number = data_base
print(type(number))
print(number)

# 等价于先把number变成一个列表，然后接受所有的剩余参数

def add(a, b, c, d):
    return a + b + c +d

# print(1, "  ", add(1, 2, 3, 4))          # 位置类参数
# print(2, "  ", add(a=1, b=2, c=3, d=4))  # 关键字类参数

args = (2, 3, 4)
kwargs = {'b': 2, 'c': 3, 'd': 4}

print(3, "  ", add(1, *args))     # 位置类参数，*位置参数
print(4, "  ", add(1, **kwargs))  # 位置类参数，**关键字参数

print(5, "  ", add(a=1, **kwargs))  # 关键字类参数，**关键字参数
# print(6, "  ", add(a=1, *args))   # 关键字类参数，*位置参数 报错


