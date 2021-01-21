from datetime import datetime
# import datetime           what is the difference
# ######################## 函数别名（指针）与调用 #########################################
def hi(name='Knight'):
    return "hi {}".format(name)
greet = hi  # 这里没有加()，因为不是调用函数
# print(greet)    # <function hi at 0x0000000002043E18>
# print(hi)       # <function hi at 0x0000000002043E18>
# print(greet())  # hi Knight
# print(hi())     # hi Knight
del hi
# print(hi())     # NameError: name 'hi' is not defined
# print(greet())    # hi Knight

# ######################## 函数作为返回对象 #########################################
def hello(name="Knight"):
    def greet_():
        return "Now U are visiting greet fun"

    def welcome():
        return "Now U are visiting welcome fun"

    if name == "Knight":
        return greet_
    else:
        return welcome

a = hello()
# print(a)    # <function hello.<locals>.greet at 0x000000000BBA7048>
# print(a())  # Now U are visiting greet fun

# ######################## 函数作为参数 #########################################
def hi(name='Knight'):
    return "hi {}".format(name)
def doSomethingBeforeHi(func):
    print("do some interesting thing before Hi")
    print(func())

# doSomethingBeforeHi(hi)  # do some interesting thing before Hi # hi Knight

# ######################## 第一个装饰器 #########################################
# 就像decorate名字一样，为函数加一些装饰的内容
def peo_info(name='Knight'):  # 被装饰的函数
    print("name: {}".format(name))

def decorate_info(func_name):
    def wrapTheFunc():
        print("school: {}".format("SEU"))
        func_name()

        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%y-%m-%d_%H:%M:%S')

        print("Time: {}".format(time_str))
    return wrapTheFunc

# peo_info()  # name: Knight
# peo_info = decorate_info(peo_info)
# peo_info()  # school: SEU # name: Knight # Time: 19-11-30_19:34:16
# print(peo_info.__name__)  # wrapTheFunc

# ############################### @ 装饰器 #########################################
@decorate_info
def soldier_info(): # 等价于在此处定义一个函数，然后把函数名作为参数传输到装饰器中
    print("soldier_name: {}".format("Knight"))

# soldier_info()  # school: SEU #　soldier_name: Knight　 # Time: 19-11-30_19:53:07, 相当于免去了前一个实验的 redundant/verbose step
# print(soldier_info.__name__)  # wrapTheFunc

# ############################### 被装饰函数__name__保持 #########################################
# 导包
# 定义一个装饰器 并用@wrap（a_fun）注释
# 定义被装饰函数
# 调用被装饰函数

from functools import wraps

def output_info(func_name):
    @ wraps(func_name)
    def more_info():
        print("位置：{}".format("东北亚"))
        func_name()
        print("首都：{}".format("北京"))
    return more_info

@output_info
def nation_info(name="China"):
    print("Nation: {}".format(name))

# nation_info()
# print(nation_info.__name__)  # nation_info

# ############################### 使用规范 #########################################
from functools import wraps
def wave_display(wave_name):
    @wraps(wave_name)
    def show_name():
        if show:
            return wave_name() + "_decorated"
        else:
            return "can not show the name"
    return show_name

show = True

@wave_display
def sine_curve():
    name = "sine_curve"
    return name

print(sine_curve())

# 相当于什么：@wave_display 等价于 sine_curve=wave_display(sine_curve)  #这就是装饰器＠的作用

