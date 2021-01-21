f = open("train.txt", encoding="UTF-8")


def fun(f):
    lst = f.read()
    lst = lst.replace("\n", "")
    lst = [w for w in lst][1:]
    return lst

res = fun(f)
print(res)

