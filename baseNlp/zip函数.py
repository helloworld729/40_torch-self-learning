a = [1, 2, 3]
b = [4, 5, 6]
zipped = zip(a, b)
print(zip(a, b)[0])
print(list(zipped))  # [(1, 4), (2, 5), (3, 6)]
print(* zip(a, b))   # (1, 4) (2, 5) (3, 6)
result = zip(*zip(a, b))
print(list(result))  # [(1, 2, 3), (4, 5, 6)]

# print(list(zipped))
# 奇怪的是用zipped替换不行

