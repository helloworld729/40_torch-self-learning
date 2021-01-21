from functools import wraps
# 带参数的装饰器,需要加括号
def log_file(name='out.log'):
    def logging_decorate(log_fun):
        @wraps(log_fun)
        def wrap_fun():
            log_str = log_fun.__name__ + " was called"
            with open(name, 'a')as f:
                f.write(log_str + '\n')
            return log_fun()
        return wrap_fun
    return logging_decorate

# @log_file()  # need parentheses
# def log1():
#     return "a happy day"
# print(log1())
#
# @log_file(name='log2.log')
# def log2():
#     pass
# log2()

# 小结：在外层又封装了一层，专门用于接受参数 即 @log_file等价于当个函数作为参数， @log_file()除了默认的函数名参数，
# 还可以添加别的参数，装饰器的层数也多了一层

# ############################################### 装饰器类 #######################################################
class logit(object):
    def __init__(self, logfile='out.log'):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function():
            log_string = func.__name__ + " was called"
            print(log_string)
            with open(self.logfile, 'a') as opened_file:
                opened_file.write(log_string + '\n')
            return func()
        return wrapped_function

# ################## 装饰器 #############################
# @logit()
def myfunc1():
    return 0
# myfunc1()
# ################## 等价形式 ############################
test = logit()
test(myfunc1)()


